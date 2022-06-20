from django.urls import path, include
from .views import interact, BotSolutions

urlpatterns = [
    path("interact/", interact, name="interact"),
    path("query/", BotSolutions.as_view(), name="query_view"),
]
