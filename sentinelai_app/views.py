import os
import tempfile
import pickle
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .models import Document, GeneratedQuestion
from .forms import DocumentUploadForm, QuestionForm, GroqAPIKeyForm
from .utils import (
    process_documents_for_rag, create_vectorstore, generate_answer,
    split_by_titles, gen_questions, bucket
)


def home(request):
    """Home page view"""
    # Count of documents and questions
    document_count = Document.objects.count()
    question_count = GeneratedQuestion.objects.count()
    
    # Get the latest API key if available
    api_key = request.session.get('groq_api_key', '')
    
    context = {
        'document_count': document_count,
        'question_count': question_count,
        'has_api_key': bool(api_key)
    }
    return render(request, 'sentinelai_app/home.html', context)


def rag_upload(request):
    """View for uploading documents for RAG"""
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save()
            
            # Process the document for RAG
            file_path = document.file.path
            file_type = document.file_type
            
            documents, chunk_count = process_documents_for_rag(file_path, file_type)
            
            if documents and chunk_count > 0:
                # Create vector store
                vectorstore = create_vectorstore(documents)
                
                if vectorstore:
                    # Save vectorstore to session
                    if 'rag_vectorstores' not in request.session:
                        request.session['rag_vectorstores'] = {}
                    
                    # Create a temporary file to store the vectorstore
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', dir=settings.MEDIA_ROOT) as f:
                        pickle.dump(vectorstore, f)
                        vs_path = os.path.basename(f.name)
                    
                    # Save the path in session
                    request.session['rag_vectorstores'][str(document.id)] = vs_path
                    request.session.modified = True
                    
                    # Mark as processed
                    document.processed = True
                    document.save()
                    
                    messages.success(request, f"Document processed successfully! Created {chunk_count} chunks.")
                else:
                    messages.error(request, "Failed to create vector store.")
            else:
                messages.error(request, "Failed to process document.")
            
            return redirect('rag_documents')
    else:
        form = DocumentUploadForm()
    
    return render(request, 'sentinelai_app/rag_upload.html', {'form': form})


def rag_documents(request):
    """View for listing documents for RAG"""
    documents = Document.objects.filter(processed=True).order_by('-uploaded_at')
    return render(request, 'sentinelai_app/rag_documents.html', {'documents': documents})


def rag_qa(request):
    """View for Q&A with RAG"""
    documents = Document.objects.filter(processed=True).order_by('-uploaded_at')
    
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            
            # Get the vectorstore paths from session
            vs_paths = request.session.get('rag_vectorstores', {})
            
            if not vs_paths:
                messages.error(request, "No documents have been processed. Please upload and process documents first.")
                return render(request, 'sentinelai_app/rag_qa.html', {'form': form, 'documents': documents})
            
            # Get API key from session
            api_key = request.session.get('groq_api_key', '')
            
            # Load each vectorstore and combine
            combined_results = []
            for doc_id, vs_path in vs_paths.items():
                full_path = os.path.join(settings.MEDIA_ROOT, vs_path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'rb') as f:
                            vectorstore = pickle.load(f)
                            answer, sources = generate_answer(question, vectorstore, api_key)
                            
                            if sources:
                                doc = Document.objects.get(id=int(doc_id))
                                combined_results.append({
                                    'document': doc.title,
                                    'answer': answer,
                                    'sources': [s.page_content[:200] + '...' for s in sources]
                                })
                    except Exception as e:
                        messages.warning(request, f"Error loading vectorstore: {str(e)}")
            
            if combined_results:
                return render(request, 'sentinelai_app/rag_qa.html', {
                    'form': form, 
                    'documents': documents,
                    'question': question,
                    'results': combined_results
                })
            else:
                messages.error(request, "Unable to generate answers. Please try again.")
    else:
        form = QuestionForm()
    
    return render(request, 'sentinelai_app/rag_qa.html', {'form': form, 'documents': documents})


def question_generation(request):
    """View for question generation from documents"""
    if 'groq_api_key' not in request.session:
        return redirect('api_key_setup')
        
    if request.method == 'POST':
        if 'document' in request.FILES:
            # Get API key from session
            api_key = request.session.get('groq_api_key')
            
            # Handle file upload
            file = request.FILES['document']
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                for chunk in file.chunks():
                    f.write(chunk)
                temp_file_path = f.name
            
            try:
                # Create document record
                document = Document.objects.create(
                    title=file.name,
                    file=file,
                    file_type='pdf'
                )
                
                # Extract sections
                sections = split_by_titles(temp_file_path)
                
                # Generate questions for each section
                all_questions = []
                for section_title, section_content in sections:
                    if len(section_content) > 100:  # Only process substantial sections
                        questions = gen_questions(section_content, api_key)
                        for q in questions:
                            all_questions.append(q)
                            GeneratedQuestion.objects.create(
                                document=document,
                                question=q,
                                section=section_title,
                                category='Uncategorized'  # Will be updated later
                            )
                
                # Categorize questions
                categorized = bucket(all_questions, api_key)
                
                # Update question categories
                for category, questions in categorized.items():
                    for q in questions:
                        GeneratedQuestion.objects.filter(
                            document=document,
                            question=q
                        ).update(category=category)
                
                messages.success(request, f"Generated {len(all_questions)} questions from {len(sections)} sections.")
                
                # Delete temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
                return redirect('view_questions', document_id=document.id)
                
            except Exception as e:
                messages.error(request, f"Error generating questions: {str(e)}")
                
                # Delete temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    return render(request, 'sentinelai_app/question_generation.html')


def view_questions(request, document_id):
    """View generated questions for a document"""
    document = get_object_or_404(Document, id=document_id)
    
    # Get questions by category
    questions_by_category = {}
    for q in document.questions.all().order_by('category', 'section'):
        if q.category not in questions_by_category:
            questions_by_category[q.category] = []
        questions_by_category[q.category].append(q)
    
    context = {
        'document': document,
        'questions_by_category': questions_by_category
    }
    
    return render(request, 'sentinelai_app/view_questions.html', context)


def api_key_setup(request):
    """View for setting up Groq API key"""
    if request.method == 'POST':
        form = GroqAPIKeyForm(request.POST)
        if form.is_valid():
            api_key = form.cleaned_data['api_key']
            request.session['groq_api_key'] = api_key
            request.session.modified = True
            
            messages.success(request, "API key saved successfully!")
            return redirect('home')
    else:
        form = GroqAPIKeyForm()
        
    return render(request, 'sentinelai_app/api_key_setup.html', {'form': form})


@csrf_exempt
def get_answer(request, question_id):
    """API endpoint for getting answer to a question"""
    if request.method == 'GET':
        question = get_object_or_404(GeneratedQuestion, id=question_id)
        
        # Get the vectorstore paths from session
        vs_paths = request.session.get('rag_vectorstores', {})
        
        if not vs_paths:
            return JsonResponse({'error': 'No documents processed for RAG'}, status=400)
        
        # Get API key from session    
        api_key = request.session.get('groq_api_key', '')
            
        # Try to generate an answer
        for doc_id, vs_path in vs_paths.items():
            full_path = os.path.join(settings.MEDIA_ROOT, vs_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'rb') as f:
                        vectorstore = pickle.load(f)
                        answer, _ = generate_answer(question.question, vectorstore, api_key)
                        
                        if answer and not answer.startswith("I'm sorry"):
                            return JsonResponse({'answer': answer})
                except Exception as e:
                    pass  # Try next vectorstore
        
        # If no good answer found
        return JsonResponse({'error': 'Could not generate a good answer'}, status=404)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
