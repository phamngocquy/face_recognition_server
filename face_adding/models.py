from django.db import models


# Create your models here.

# Person
class Person(models.Model):
    name = models.CharField(max_length=50, unique=True)
    create_time = models.DateTimeField(auto_now_add=True, editable=False, null=False, blank=False)
    update_time = models.DateTimeField(auto_now=True, editable=False, null=False, blank=False)


class Image(models.Model):
    path = models.TextField(null=False, blank=False, unique=True)
    aligned = models.BooleanField(default=False)
    create_time = models.DateTimeField(auto_now_add=True, editable=False, null=False, blank=False)
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='person_id_image')


class ImageAligned(models.Model):
    path = models.TextField(null=False, blank=False, unique=True)
    create_time = models.DateTimeField(auto_now_add=True, editable=False, null=False, blank=False)
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name='image_aligned_id_image')
