const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://backend:8000";

export async function getHealth(): Promise<unknown> {
  const response = await fetch(`${apiBaseUrl}/health`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

export async function getRuntimeOptions(): Promise<unknown> {
  const response = await fetch(`${apiBaseUrl}/runtime/options`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Runtime options request failed: ${response.status}`);
  }
  return response.json();
}
