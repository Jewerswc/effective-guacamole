# webhook_handlers.py
import djstripe.models
from djstripe import webhooks

@webhooks.handler("checkout.session.completed")
def handle_checkout_session_completed(event, **kwargs):
    session = event.data['object']
    customer_email = session.get('customer_email')
    subscription_id = session.get('subscription')
    
    if customer_email and subscription_id:
        try:
            user = User.objects.get(email=customer_email)
            user.subscription_id = subscription_id
            user.is_subscribed = True
            user.save()
        except User.DoesNotExist:
            pass
