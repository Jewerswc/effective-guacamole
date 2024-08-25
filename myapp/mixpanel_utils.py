from mixpanel import Mixpanel
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Mixpanel with your token
mp = Mixpanel(settings.MIXPANEL_TOKEN)

def is_elb_health_check(user_agent, ip_address):
    """
    Detect if the request is from an ELB Health Checker.

    :param user_agent: The user agent string.
    :param ip_address: The IP address of the user.
    :return: True if it is an ELB Health Checker, False otherwise.
    """
    return "ELB-HealthChecker/2.0" in user_agent or ip_address == "127.0.0.1"

def track_event(event_name, distinct_id, properties=None):
    """
    Track an event with Mixpanel.

    :param event_name: Name of the event.
    :param distinct_id: Unique identifier for the user/session.
    :param properties: Optional properties for the event.
    """
    if properties is None:
        properties = {}

    user_agent = properties.get('user_agent', '')
    ip_address = properties.get('ip_address', '')

    # Skip tracking for ELB health checks
    if is_elb_health_check(user_agent, ip_address):
        logger.info(f"Skipping ELB health check traffic for distinct_id: {distinct_id} with user agent: {user_agent} and IP: {ip_address}")
        return

    logger.info(f"Tracking event: {event_name} for distinct_id: {distinct_id} with properties: {properties}")
    mp.track(distinct_id, event_name, properties)

# Example usage
# track_event("User Signup", "unique_user_123", {"user_agent": "Mozilla/5.0", "ip_address": "8.8.8.8"})
