/** @type {import('next').NextConfig} */
function normalizeApiBaseUrl(baseUrl) {
  return baseUrl.replace(/\/+$/, '') || baseUrl;
}

const apiBaseUrl = normalizeApiBaseUrl(process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api');

const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${apiBaseUrl}/:path*`,
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/api/backtest/run-all-stream',
        headers: [
          { key: 'Cache-Control', value: 'no-cache, no-transform' },
          { key: 'X-Accel-Buffering', value: 'no' },
          { key: 'Content-Type', value: 'text/event-stream' },
          { key: 'Connection', value: 'keep-alive' },
          { key: 'Transfer-Encoding', value: 'chunked' },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
