from django.utils.deprecation import MiddlewareMixin
import logging

logger = logging.getLogger(__name__)

class TrackVisitorMiddleware(MiddlewareMixin):
    def get_client_ip(self, request):
        """
        Get the real client IP address from the request, considering possible headers set by the load balancer.
        
        :param request: Django request object.
        :return: Real client IP address.
        """
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

    def process_request(self, request):
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        ip_address = self.get_client_ip(request)

        # Attach user agent and IP address to the request for tracking purposes
        request.user_agent = user_agent
        request.ip_address = ip_address

        # Log the request details for debugging
        logger.debug(f"Request details - User Agent: {user_agent}, IP Address: {ip_address}")

        # Add any other tracking logic here

        return None

# Update MIDDLEWARE in settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'myapp.middleware.TrackVisitorMiddleware',
    'django.middleware.http.ConditionalGetMiddleware',
    'django.middleware.gzip.GZipMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.common.BrokenLinkEmailsMiddleware',
    'django.middleware.locale.LocaleMiddleware',
]
