import uuid
from .mixpanel_utils import track_event

class TrackVisitorMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Generate a unique identifier for the session if it doesn't exist
        if not request.session.get('distinct_id'):
            request.session['distinct_id'] = str(uuid.uuid4())
        
        distinct_id = request.session['distinct_id']
        
        # Track the visit
        track_event('Site Visit', distinct_id, {
            'path': request.path,
            'method': request.method,
            'user_agent': request.META.get('HTTP_USER_AGENT', ''),
            'ip_address': request.META.get('REMOTE_ADDR', '')
        })

        return response
