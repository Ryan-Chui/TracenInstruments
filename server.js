const http = require('http');
const fs = require('fs');
const path = require('path');

const hostname = '127.0.0.1';
const port = 3000;

const FLASK_HOST = '127.0.0.1';
const FLASK_PORT = 5000;

const server = http.createServer((req, res) => {
    const url = new URL(req.url, `http://${hostname}:${port}`);
    const pathname = url.pathname;

    // Forward face scanner requests to Flask.
    if (pathname === '/health' || pathname === '/video_feed') {
        const proxyReq = http.request(
            {
                hostname: FLASK_HOST,
                port: FLASK_PORT,
                path: url.pathname + url.search,
                method: req.method,
                headers: {
                    accept: req.headers.accept || '*/*'
                }
            },
            (proxyRes) => {
                res.writeHead(proxyRes.statusCode || 502, {
                    'Content-Type':
                        proxyRes.headers['content-type'] ||
                        'application/json',
                    'Cache-Control': 'no-store'
                });

                // Forward video frames as they arrive.
                proxyRes.pipe(res);
            }
        );

        proxyReq.on('error', (err) => {
            console.error('Flask proxy error:', err.message);

            if (!res.headersSent) {
                res.writeHead(502, {
                    'Content-Type': 'application/json'
                });

                res.end(JSON.stringify({
                    ok: false,
                    error:
                        'Cannot connect to Flask. Make sure python app.py is running on port 5000.'
                }));
            } else {
                res.destroy();
            }
        });

        // Stop the upstream request if the browser disconnects.
        res.on('close', () => {
            if (!res.writableEnded) {
                proxyReq.destroy();
            }
        });

        proxyReq.end();
        return;
    }

    // Serve the existing HTML pages.
    let filePath;

    switch (pathname) {
        case '/':
            filePath = path.join(__dirname, 'public', 'index.html');
            break;

        case '/shop':
            filePath = path.join(__dirname, 'public', 'shop.html');
            break;

        default:
            res.statusCode = 404;
            res.setHeader('Content-Type', 'text/plain; charset=utf-8');
            res.end('404 - Page not found');
            return;
    }

    fs.readFile(filePath, (err, data) => {
        if (err) {
            console.error('File read error:', err.message);
            res.statusCode = 500;
            res.end('Could not load the requested page.');
            return;
        }

        res.statusCode = 200;
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.end(data);
    });
});

server.listen(port, hostname, () => {
    console.log(`Node.js server running at http://${hostname}:${port}`);
});