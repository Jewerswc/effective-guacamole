from django.contrib import admin
from django.urls import path
from .views import (
    SubscribeAPIView, ArticlesViewSet, SignupAPIView, LoginAPIView, health_check,
    LikeArticleAPIView, DislikeArticleAPIView, RankUserArticlesAPIView,
    user_preferences, submit_article, article_detail, RssFeedListCreateAPIView, ArticlesListView, add_rss_feed, user_articles, RssFeedDestroyAPIView
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('like_article/', LikeArticleAPIView.as_view(), name='like_article'),
    path('dislike_article/', DislikeArticleAPIView.as_view(), name='dislike_article'),
    path('rank_articles/', RankUserArticlesAPIView.as_view(), name='rank_articles'),
    path('subscribe/', SubscribeAPIView.as_view(), name='subscribe'),
    path('articles/', ArticlesViewSet.as_view({'get': 'list', 'post': 'create'}), name='articles'),
    path('signup/', SignupAPIView.as_view(), name='signup'),
    path('login/', LoginAPIView.as_view(), name='login'),
    path('preferences/', user_preferences, name='preferences'),
    path('health/', health_check, name='health_check'),
    path('submit-article/', submit_article, name='submit_article'),
    path('article/<slug:slug>/', article_detail, name='article_detail'),
    path('rss-feeds/', RssFeedListCreateAPIView.as_view(), name='rss_feed_list_create'),
    path('articlesb/', ArticlesListView.as_view(), name='articles_list'),
    path('add-rss-feed/', add_rss_feed, name='add_rss_feed'),
    path('user-articles/', user_articles, name='user_articles'),
    path('rss-feeds/<int:pk>/', RssFeedDestroyAPIView.as_view(), name='rss-feed-destroy'),
]
