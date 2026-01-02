from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class StripPathMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # request.scope["path"]는 str, raw_path는 bytes
        path = request.scope.get("path")
        if isinstance(path, str):
            # 오른쪽(끝) 공백/개행 제거
            request.scope["path"] = path.rstrip(" \t\r\n")
        raw = request.scope.get("raw_path")
        if isinstance(raw, (bytes, bytearray)):
            request.scope["raw_path"] = raw.rstrip(b" \t\r\n")
        return await call_next(request)
