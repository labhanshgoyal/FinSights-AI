from supabase import create_client
import streamlit as st

def get_auth_client():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

def sign_up(email, password):
    try:
        supabase = get_auth_client()
        result = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        if result.user:
            return {"success": True, "user": result.user}
        return {"success": False, "error": "Signup failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def sign_in(email, password):
    try:
        supabase = get_auth_client()
        result = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if result.user:
            return {
                "success": True,
                "user": result.user,
                "session": result.session
            }
        return {"success": False, "error": "Invalid credentials"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_oauth_url(provider):
    try:
        supabase = get_auth_client()
        result = supabase.auth.sign_in_with_oauth({
            "provider": provider,
            "options": {
                "redirect_to": st.secrets.get("REDIRECT_URL", "http://localhost:8501")
            }
        })
        return result.url
    except Exception:
        return None

def get_current_user():
    return st.session_state.get("user", None)

def logout():
    for key in ["user", "user_id", "user_email", "messages"]:
        if key in st.session_state:
            del st.session_state[key]