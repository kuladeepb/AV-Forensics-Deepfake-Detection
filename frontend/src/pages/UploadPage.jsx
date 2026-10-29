import { useState, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Film, AlertCircle, Loader2, ServerCrash } from 'lucide-react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || ''
const MAX_MB  = 50
const ALLOWED_EXTS = ['.mp4', '.mov', '.avi']

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export default function UploadPage() {
  const navigate = useNavigate()
  const inputRef  = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [file, setFile]         = useState(null)
  const [error, setError]       = useState(null)
  const [status, setStatus]     = useState('idle') // idle | uploading | waking | error
  const [progress, setProgress] = useState(0)

  const validateFile = (f) => {
    const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase()
    if (!ALLOWED_EXTS.includes(ext)) {
      return `Unsupported file type "${ext}". Allowed: ${ALLOWED_EXTS.join(', ')}`
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      return `File too large (${formatBytes(f.size)}). Maximum is ${MAX_MB} MB.`
    }
    return null
  }

  const selectFile = useCallback((f) => {
    setError(null)
    const err = validateFile(f)
    if (err) { setError(err); return }
    setFile(f)
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) selectFile(f)
  }, [selectFile])

  const onInputChange = (e) => {
    const f = e.target.files[0]
    if (f) selectFile(f)
  }

  const submit = async () => {
    if (!file) return
    setStatus('uploading')
    setProgress(0)
    setError(null)

    const form = new FormData()
    form.append('file', file)

    // Wake-up indicator for cold-start free hosting
    const wakeTimer = setTimeout(() => setStatus('waking'), 8000)

    try {
      const res = await axios.post(`${API_URL}/api/analyze`, form, {
        timeout: 120_000,
        onUploadProgress: (e) => {
          setProgress(Math.round((e.loaded / e.total) * 100))
        },
      })
      clearTimeout(wakeTimer)
      navigate('/result', { state: res.data })
    } catch (err) {
      clearTimeout(wakeTimer)
      setStatus('error')
      const msg = err.response?.data?.detail || err.message || 'Upload failed.'
      setError(msg)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-16">
      {/* Hero */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-white mb-3">
          Audio-Visual Deepfake Detection
        </h1>
        <p className="text-gray-400 text-lg">
          Upload a video to check lip-sync consistency and detect AI manipulation.
          Receive a downloadable forensic report.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-12
          flex flex-col items-center gap-4 transition-all duration-200
          ${dragging
            ? 'border-brand-500 bg-brand-600/10 scale-[1.01]'
            : file
            ? 'border-green-600 bg-green-600/5'
            : 'border-gray-700 bg-gray-900 hover:border-gray-500 hover:bg-gray-800/50'
          }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,.avi"
          className="hidden"
          onChange={onInputChange}
        />

        {file ? (
          <>
            <Film className="w-12 h-12 text-green-400" />
            <div className="text-center">
              <p className="text-white font-semibold text-lg">{file.name}</p>
              <p className="text-gray-400 text-sm mt-1">{formatBytes(file.size)}</p>
            </div>
            <p className="text-gray-500 text-sm">Click or drop to change file</p>
          </>
        ) : (
          <>
            <Upload className="w-12 h-12 text-gray-500" />
            <div className="text-center">
              <p className="text-white font-semibold text-lg">Drop your video here</p>
              <p className="text-gray-400 text-sm mt-1">
                or click to browse — MP4, MOV, AVI · max {MAX_MB} MB
              </p>
            </div>
          </>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 flex items-start gap-3 p-4 rounded-xl bg-danger-600/10 border border-danger-600/30 text-danger-500">
          <AlertCircle className="w-5 h-5 mt-0.5 shrink-0" />
          <p className="text-sm">{error}</p>
        </div>
      )}

      {/* Upload progress */}
      {(status === 'uploading' || status === 'waking') && (
        <div className="mt-4 space-y-2">
          {status === 'uploading' && (
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-brand-500 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}
          <div className="flex items-center gap-2 text-gray-400 text-sm">
            {status === 'waking' ? (
              <>
                <ServerCrash className="w-4 h-4 text-amber-400" />
                <span>Waking up the server (free hosting cold start, ~30 s)…</span>
              </>
            ) : (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Uploading… {progress}%</span>
              </>
            )}
          </div>
        </div>
      )}

      {/* CTA */}
      <div className="mt-6 flex justify-center">
        <button
          onClick={submit}
          disabled={!file || status === 'uploading' || status === 'waking'}
          className="btn-primary text-base px-10 py-4"
        >
          {status === 'uploading' || status === 'waking' ? (
            <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing…</>
          ) : (
            <><Upload className="w-5 h-5" /> Analyze Video</>
          )}
        </button>
      </div>

      {/* Note */}
      <p className="text-center text-gray-600 text-xs mt-6">
        Videos are deleted immediately after processing. Results cached ~1 hour.
      </p>
    </div>
  )
}

