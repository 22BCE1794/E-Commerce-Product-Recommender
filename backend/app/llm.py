import os
from dotenv import load_dotenv
load_dotenv()
import openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
LLM_MOCK = os.getenv('LLM_MOCK', 'false').lower() in ('1','true','yes')
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def explain_simple_prompt(user_recent, product, signals, tone='friendly'):
    recent = ', '.join(user_recent[:3]) if user_recent else 'your recent activity'
    attrs = product.get('attributes') or {}
    feat = ''
    if isinstance(attrs, dict) and len(attrs)>0:
        k = next(iter(attrs))
        feat = f" It has {k} = {attrs[k]}."
    return f"Because you recently looked at {recent}, we recommend {product.get('title')} — {signals.get('score','similar')}.{feat}"

def generate_explanation(user_recent, product, signals, tone='friendly'):
    if LLM_MOCK or not OPENAI_API_KEY:
        return explain_simple_prompt(user_recent, product, signals, tone)
    # Call OpenAI ChatCompletion (gpt-4o-mini is used as example — you can change model)
    prompt = f"""User recent products: {user_recent}
Product: {product.get('title')}
Product attributes: {product.get('attributes')}
Signals: {signals}
Write a concise, friendly 1-2 sentence explanation why this product is recommended to the user, referencing only the facts above."""
    try:
        res = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[{'role':'user','content':prompt}],
            max_tokens=80,
            temperature=0.2
        )
        text = res['choices'][0]['message']['content'].strip()
        return text
    except Exception as e:
        return explain_simple_prompt(user_recent, product, signals, tone)
