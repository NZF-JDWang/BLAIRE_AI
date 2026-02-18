import { NextRequest } from "next/server";

const INTERNAL_API_BASE_URL = process.env.INTERNAL_API_BASE_URL ?? "http://backend:8000";

async function proxy(request: NextRequest, params: { path: string[] }) {
  const target = new URL(`${INTERNAL_API_BASE_URL}/${params.path.join("/")}`);
  request.nextUrl.searchParams.forEach((value, key) => target.searchParams.set(key, value));

  const headers = new Headers();
  headers.set("accept", request.headers.get("accept") ?? "application/json");
  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  const requestApiKey = request.headers.get("x-api-key");
  const cookieApiKey = request.cookies.get("blaire_api_key")?.value;
  const effectiveApiKey = requestApiKey || cookieApiKey;
  if (effectiveApiKey) {
    headers.set("x-api-key", effectiveApiKey);
  } else {
    return new Response(JSON.stringify({ detail: "Missing API key" }), {
      status: 401,
      headers: { "content-type": "application/json" },
    });
  }

  const init: RequestInit = {
    method: request.method,
    headers,
    body:
      request.method === "GET" || request.method === "HEAD"
        ? undefined
        : Buffer.from(await request.arrayBuffer()),
    redirect: "manual",
  };

  const upstream = await fetch(target, init);
  const upstreamHeaders = new Headers(upstream.headers);
  upstreamHeaders.delete("content-encoding");
  upstreamHeaders.delete("content-length");
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: upstreamHeaders,
  });
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, await context.params);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, await context.params);
}

export async function PUT(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, await context.params);
}

export async function PATCH(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, await context.params);
}

export async function DELETE(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, await context.params);
}
