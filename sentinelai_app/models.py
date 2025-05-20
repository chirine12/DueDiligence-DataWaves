from django.db import models

# Create your models here.

class Document(models.Model):
    """Model for uploaded documents"""
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    file_type = models.CharField(max_length=10)  # pdf, txt, docx

    def __str__(self):
        return self.title


class GeneratedQuestion(models.Model):
    """Model for questions generated from documents"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='questions')
    question = models.TextField()
    section = models.CharField(max_length=255)
    category = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question
