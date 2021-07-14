from django.db import models

# Create your models here.

class CurrentInput(models.Model):
    contents = models.TextField()
    def __str__(self):
        return str(self.contents)