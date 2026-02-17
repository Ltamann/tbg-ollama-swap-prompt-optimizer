package proxy

import (
	"net/http"
	"strconv"
	"strings"
)

// selectEncoding chooses the best encoding based on Accept-Encoding header
// Returns the encoding ("br", "gzip", or "") and the corresponding file extension
func selectEncoding(acceptEncoding string) (encoding, ext string) {
	if acceptEncoding == "" {
		return "", ""
	}

	// Prefer brotli whenever it is listed at all. This keeps behavior stable
	// across clients that send weighted encodings in different orders.
	for _, part := range strings.Split(acceptEncoding, ",") {
		token := strings.TrimSpace(part)
		if token == "" {
			continue
		}
		pieces := strings.Split(token, ";")
		enc := strings.TrimSpace(pieces[0])
		if enc != "br" {
			continue
		}
		q := 1.0
		if len(pieces) > 1 {
			for _, p := range pieces[1:] {
				p = strings.TrimSpace(p)
				if !strings.HasPrefix(strings.ToLower(p), "q=") {
					continue
				}
				if parsed, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(p, "q=")), 64); err == nil {
					q = parsed
				}
			}
		}
		if q > 0 {
			return "br", ".br"
		}
	}

	for _, part := range strings.Split(acceptEncoding, ",") {
		token := strings.TrimSpace(part)
		if token == "" {
			continue
		}
		pieces := strings.Split(token, ";")
		enc := strings.TrimSpace(pieces[0])
		if enc != "gzip" {
			continue
		}
		q := 1.0
		if len(pieces) > 1 {
			for _, p := range pieces[1:] {
				p = strings.TrimSpace(p)
				if !strings.HasPrefix(strings.ToLower(p), "q=") {
					continue
				}
				if parsed, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(p, "q=")), 64); err == nil {
					q = parsed
				}
			}
		}
		if q > 0 {
			return "gzip", ".gz"
		}
	}

	return "", ""
}

// ServeCompressedFile serves a file with compression support.
// It checks for pre-compressed versions and serves them with proper headers.
func ServeCompressedFile(fs http.FileSystem, w http.ResponseWriter, r *http.Request, name string) {
	encoding, ext := selectEncoding(r.Header.Get("Accept-Encoding"))

	// Try to serve compressed version if client supports it
	if encoding != "" {
		if cf, err := fs.Open(name + ext); err == nil {
			defer cf.Close()

			// Verify it's a regular file (not a directory)
			if stat, err := cf.Stat(); err == nil && !stat.IsDir() {
				// Set the content encoding header
				w.Header().Set("Content-Encoding", encoding)
				w.Header().Add("Vary", "Accept-Encoding")

				// Get original file info for content type detection
				origFile, err := fs.Open(name)
				if err == nil {
					origFile.Close()
				}

				// Serve the compressed file
				http.ServeContent(w, r, name, stat.ModTime(), cf)
				return
			}
		}
	}

	// Fall back to serving the uncompressed file
	file, err := fs.Open(name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if stat.IsDir() {
		http.Error(w, "is a directory", http.StatusForbidden)
		return
	}

	http.ServeContent(w, r, name, stat.ModTime(), file)
}
