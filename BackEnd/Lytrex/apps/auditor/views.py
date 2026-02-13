from django.shortcuts import render
from django.http import JsonResponse


# Create your views here.
def landing(request):
    return JsonResponse({"message": "audit amazing"})