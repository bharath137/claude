import argparse
import json
import os
import secrets
import sys
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
CLAUDE_MODEL = "claude-sonnet-5-5"
LINKEDIN_VERSION = "202608"
REDIRECT_URI = "http://localhost:8000/callback"
SCOPES = "openid profile w_member_social"
TOKEN_FILE = Path(".linkedin_token.json")

CLIENT_ID = os.environ.get("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.environ.get("LINKEDIN_CLIENT_SECRET")


# ---------- 1. Generate the post with Claude ----------
def generate_post(topic: str) -> str:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=(
            "You write LinkedIn posts for an experienced software developer. "
            "Voice: practical, first person, no hype, no emojis unless natural. "
            "Format: a strong one-line hook, short paragraphs, 120-220 words, "
            "end with a question to invite comments, then 2-3 relevant hashtags. "
            "Return ONLY the post text, no preamble."
        ),
        messages=[{"role": "user", "content": f"Write a LinkedIn post about: {topic}"}],
    )
    text_blocks = [block.text for block in message.content if block.type == "text"]
    if not text_blocks:
        sys.exit(f"Claude returned no text (stop_reason: {message.stop_reason})")
    return "\n".join(text_blocks).strip()


# ---------- 2. LinkedIn OAuth ----------
class _CallbackHandler(BaseHTTPRequestHandler):
    query = {}

    def do_GET(self):
        _CallbackHandler.query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>LinkedIn authorised. You can close this tab.</h2>")

    def log_message(self, *args):
        pass


def _authorise() -> dict:
    state = secrets.token_urlsafe(16)
    auth_url = "https://www.linkedin.com/oauth/v2/authorization?" + urllib.parse.urlencode({
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
    })
    print("Opening browser to sign in to LinkedIn...")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", 8000), _CallbackHandler)
    server.handle_request()
    q = _CallbackHandler.query

    if q.get("state", [None])[0] != state:
        sys.exit("OAuth state mismatch — aborting.")
    if "error" in q:
        sys.exit(f"LinkedIn returned an error: {q.get('error_description', q['error'])}")

    r = requests.post(
        "https://www.linkedin.com/oauth/v2/accessToken",
        data={
            "grant_type": "authorization_code",
            "code": q["code"][0],
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    r.raise_for_status()
    token = r.json()
    token["expires_at"] = time.time() + token["expires_in"]
    TOKEN_FILE.write_text(json.dumps(token))
    return token


def get_access_token() -> str:
    if TOKEN_FILE.exists():
        token = json.loads(TOKEN_FILE.read_text())
        if token["expires_at"] - time.time() > 3600:  # still valid for > 1 hour
            return token["access_token"]
    return _authorise()["access_token"]


def get_person_urn(access_token: str) -> str:
    r = requests.get(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    r.raise_for_status()
    return f"urn:li:person:{r.json()['sub']}"


# ---------- 3. Publish ----------
def escape_little_text(text: str) -> str:
    """LinkedIn's 'commentary' field treats these characters as markup.
    Unescaped, they can silently truncate your post (e.g. at the first '(')."""
    for ch in "\\|{}@[]()<>#*_~":
        text = text.replace(ch, "\\" + ch)
    return text


def publish_post(access_token: str, author_urn: str, text: str) -> str:
    body = {
        "author": author_urn,
        "commentary": escape_little_text(text),
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    r = requests.post(
        "https://api.linkedin.com/rest/posts",
        headers={
            "Authorization": f"Bearer {access_token}",
            "LinkedIn-Version": LINKEDIN_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )
    if r.status_code != 201:
        sys.exit(f"LinkedIn post failed ({r.status_code}): {r.text}")
    return r.headers.get("x-restli-id", "")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Generate and post to LinkedIn with Claude.")
    parser.add_argument("topic", help="What the post should be about")
    parser.add_argument("--yes", action="store_true", help="Post without asking for confirmation")
    args = parser.parse_args()

    if not (CLIENT_ID and CLIENT_SECRET and os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("Missing keys — check your .env file.")

    post = generate_post(args.topic)
    print("\n" + "=" * 60 + "\n" + post + "\n" + "=" * 60 + "\n")

    if not args.yes and input("Publish this to LinkedIn? [y/N] ").strip().lower() != "y":
        print("Not posted.")
        return

    token = get_access_token()
    urn = get_person_urn(token)
    post_id = publish_post(token, urn, post)
    print(f"Posted! https://www.linkedin.com/feed/update/{post_id}/")


if __name__ == "__main__":
    main()
