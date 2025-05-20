from django import template

register = template.Library()

@register.filter
def addclass(field, css_class):
    """Add a CSS class to a Django form field"""
    return field.as_widget(attrs={"class": css_class}) 