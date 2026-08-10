from supabase import create_client
import streamlit as st
from datetime import datetime

def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

def save_chat(user_id, ticker, role, content, quality_score=None):
    try:
        supabase = get_supabase()
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "ticker":ticker,
            "role":role,
            "content":content,
            "quality_score":quality_score
        }).execute()
    except Exception:
        pass

def get_chat_history(user_id, ticker, limit=50):
    try:
        supabase = get_supabase()
        result = supabase.table("chat_history") \
            .select("role, content, quality_score, created_at") \
            .eq("user_id", user_id) \
            .eq("ticker", ticker) \
            .order("created_at", desc=False) \
            .limit(limit) \
            .execute()
        return result.data
    except Exception:
        return []

def log_query(user_id, ticker, query, response_time_ms=None, quality_score=None):
    try:
        supabase=get_supabase()
        supabase.table("query_logs").insert({
            "user_id": user_id,
            "ticker": ticker,
            "query": query,
            "response_time_ms": response_time_ms,
            "quality_score": quality_score
        }).execute()
    except Exception:
        pass

def get_query_stats(user_id):
    try:
        supabase = get_supabase()
        result = supabase.table("query_logs") \
            .select("ticker, quality_score, response_time_ms, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(100) \
            .execute()
        return result.data
    except Exception:
        return []

def add_to_watchlist(user_id, ticker):
    try:
        supabase = get_supabase()
        supabase.table("watchlists").insert({
            "user_id": user_id,
            "ticker": ticker
        }).execute()
        return True
    except Exception:
        return False
        
def remove_from_watchlist(user_id, ticker):
    try:
        supabase = get_supabase()
        supabase.table("watchlists") \
            .delete() \
            .eq("user_id", user_id) \
            .eq("ticker", ticker) \
            .execute()
        return True
    except Exception:
        return False
        
def get_watchlist(user_id):
    try:
        supabase = get_supabase()
        result = supabase.table("watchlists") \
            .select("ticker, added_at") \
            .eq("user_id", user_id) \
            .order("added_at", desc=True) \
            .execute()
        return [item["ticker"] for item in result.data]
    except Exception:
        return []