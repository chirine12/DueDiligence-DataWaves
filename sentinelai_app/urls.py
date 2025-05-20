from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    
    # RAG (Document Upload & Q&A)
    path('rag/upload/', views.rag_upload, name='rag_upload'),
    path('rag/documents/', views.rag_documents, name='rag_documents'),
    path('rag/qa/', views.rag_qa, name='rag_qa'),
    
    # Question Generation
    path('question-generation/', views.question_generation, name='question_generation'),
    path('questions/<int:document_id>/', views.view_questions, name='view_questions'),
    
    # API Key Setup
    path('api-key-setup/', views.api_key_setup, name='api_key_setup'),
    
    # API Endpoints
    path('api/get-answer/<int:question_id>/', views.get_answer, name='get_answer'),
] 