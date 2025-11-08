from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('login', views.login, name="login"),
    path('logout', views.logout, name="logout"),
    path('register', views.register, name="register"),
    path('dashboard', views.dashboard, name="dashboard"),
    
    path('portfolio', views.portfolio, name='portfolio'),
    path('news', views.news, name='news'),
    path('market', views.market, name='market'),
    path("add_to_wishlist/", views.add_to_wishlist, name="add_to_wishlist"),
    path('remove_from_wishlist/<str:symbol>/', views.remove_from_wishlist, name='remove_from_wishlist'),
]
