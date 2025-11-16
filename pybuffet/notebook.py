from IPython.display import HTML, display
import markdown

def display_markdown_as_html(content):
    html = markdown.markdown(content)
    display(HTML(html))
    
def display_html(content, selector='table'):
    if content.startswith('```html'):
        content = content[len('```html'):].rstrip('```').strip()
    css = f"<style>{selector} th, {selector} td {{ text-align:left !important; }}</style>"
    display(HTML(css + content))
    
__all__ = ['display_markdown_as_html', 'display_html']