from django import forms
from .models import Document


class DocumentUploadForm(forms.ModelForm):
    """Form for uploading documents for RAG system"""
    class Meta:
        model = Document
        fields = ('title', 'file')
        
    def clean_file(self):
        file = self.cleaned_data.get('file')
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension not in ['pdf', 'txt', 'docx']:
            raise forms.ValidationError("Only PDF, TXT, and DOCX files are allowed.")
        
        # Save the file type for later use
        self.instance.file_type = file_extension
        
        return file


class QuestionForm(forms.Form):
    """Form for asking questions to the RAG system"""
    question = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': 'Ask a question about your documents...'}),
        label="Your Question"
    )


class GroqAPIKeyForm(forms.Form):
    """Form for entering Groq API Key"""
    api_key = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Enter your Groq API Key'}),
        required=True,
        label="Groq API Key"
    ) 