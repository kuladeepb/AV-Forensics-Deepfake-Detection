import { useLocation, useNavigate } from 'react-router-dom'
import { useRef, useState, useCallback } from 'react'
import {
  CheckCircle, XCircle, Download, ChevronLeft,
  AlertTriangle, AudioLines, Video, Layers, Shield,
} from 'lucide-react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, ReferenceLine, ReferenceArea, CartesianGrid,
} from 'recharts'

const API_URL = import.meta.env.VITE_API_URL || ''

const MODALITY_ICONS = {
  'Authentic':          Shield,
  'Audio Manipulated':  AudioLines,
  'Video Manipulated':  Video,
  'Both Manipulated':   Layers,
}

const MODALITY_COLORS = {
  'Authentic':          'text-green-400 bg-green-400/10 border-green-500/30',
  'Audio Manipulated':  'text-amber-400 bg-amber-400/10 border-amber-500/30',
  'Video Manipulated':  'text-orange-400 bg-orange-400/10 border-orange-500/30',
  'Both Manipulated':   'text-red-400 bg-red-400/10 border-red-500/30',
}

function ConfidenceBar({ value, isFake }) {
  return (
    <div className="mt-3">
      <div className="flex justify-between text-sm text-gray-400 mb-1">
        <span>Confidence</span>
        <span className="font-mono font-medium text-white">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            isFake ? 'bg-red-500' : 'bg-green-500'
          }`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )
}

function SyncChart({ timeline, suspects, videoRef }) {
  const [hoveredT, setHoveredT] = useState(null)

  const handleClick = useCallback((data) => {
    if (data?.activePayload?.[0] && videoRef.current) {
      videoRef.current.currentTime = data.activePayload[0].payload.t
    }
  }, [videoRef])

  return (
    <div className="h-52">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={timeline}
          margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
          onClick={handleClick}
          style={{ cursor: 'pointer' }}
          onMouseMove={(e) => {
            if (e?.activePayload?.[0]) setHoveredT(e.activePayload[0].payload.t)
          }}
          onMouseLeave={() => setHoveredT(null)}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          {suspects.map((seg, i) => (
            <ReferenceArea
              key={i}
              x1={seg.start} x2={seg.end}
              fill="#EF4444" fillOpacity={0.12}
            />
          ))}
          <XAxis
            dataKey="t"
            tickFormatter={(v) => `${v.toFixed(1)}s`}
            stroke="#6B7280"
            tick={{ fill: '#9CA3AF', fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            stroke="#6B7280"
            tick={{ fill: '#9CA3AF', fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{
              background: '#111827', border: '1px solid #374151',
              borderRadius: 8, color: '#F9FAFB', fontSize: 12,
            }}
            formatter={(v) => [`${v.toFixed(4)}`, 'Sync Score']}
            labelFormatter={(l) => `t = ${parseFloat(l).toFixed(2)} s`}
          />
          <ReferenceLine y={0.5} stroke="#F59E0B" strokeDasharray="4 4" label={{ value: 'Threshold', fill: '#F59E0B', fontSize: 10 }} />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#60A5FA' }}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-600 mt-1">Click chart to seek video to that timestamp</p>
    </div>
  )
}

export default function ResultPage() {
  const { state: data } = useLocation()
  const navigate  = useNavigate()
  const videoRef  = useRef(null)
  const [downloading, setDownloading] = useState(false)

  if (!data) {
    return (
      <div className="max-w-xl mx-auto px-4 py-20 text-center">
        <p className="text-gray-400">No result found. Please upload a video first.</p>
        <button onClick={() => navigate('/')} className="btn-primary mt-6">
          <ChevronLeft className="w-4 h-4" /> Back to Upload
        </button>
      </div>
    )
  }

  const isFake       = data.verdict === 'FAKE'
  const mod          = data.modality || {}
  const modLabel     = mod.label || 'Unknown'
  const ModIcon      = MODALITY_ICONS[modLabel] || Shield
  const modColor     = MODALITY_COLORS[modLabel] || 'text-gray-400 bg-gray-800 border-gray-700'
  const timeline     = data.sync_timeline || []
  const suspects     = data.suspect_segments || []
  const evidences    = data.evidence_frames || []

  const downloadReport = async () => {
    setDownloading(true)
    try {
      const res = await fetch(`${API_URL}/api/report/${data.case_id}`)
      if (!res.ok) throw new Error('Report unavailable.')
      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `forensic_report_${data.case_id.slice(0, 8)}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      alert(`Download failed: ${err.message}`)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-6">
      {/* Back + Download */}
      <div className="flex items-center justify-between">
        <button onClick={() => navigate('/')} className="btn-secondary text-sm">
          <ChevronLeft className="w-4 h-4" /> New Analysis
        </button>
        <button onClick={downloadReport} disabled={downloading} className="btn-primary text-sm">
          <Download className="w-4 h-4" />
          {downloading ? 'Generating…' : 'Download Forensic Report'}
        </button>
      </div>

      {/* Verdict + Modality */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Verdict card */}
        <div className={`card ${isFake ? 'border-red-600/40' : 'border-green-600/40'}`}>
          <div className="flex items-center gap-3 mb-2">
            {isFake
              ? <XCircle className="w-8 h-8 text-red-500" />
              : <CheckCircle className="w-8 h-8 text-green-400" />
            }
            <div>
              <p className="text-gray-400 text-sm">Verdict</p>
              <p className={`text-4xl font-bold ${isFake ? 'text-red-400' : 'text-green-400'}`}>
                {data.verdict}
              </p>
            </div>
          </div>
          <ConfidenceBar value={data.confidence} isFake={isFake} />
          <p className="text-gray-500 text-xs mt-3">
            Case: <span className="font-mono">{data.case_id}</span>
          </p>
          {data._mock && (
            <div className="mt-3 flex items-center gap-2 text-amber-400 text-xs">
              <AlertTriangle className="w-3.5 h-3.5" />
              Demo mode — mock result (no trained weights loaded)
            </div>
          )}
        </div>

        {/* Modality card */}
        <div className="card">
          <p className="text-gray-400 text-sm mb-3">Manipulated Modality</p>
          <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border ${modColor} mb-4`}>
            <ModIcon className="w-5 h-5" />
            <span className="font-semibold">{modLabel}</span>
          </div>

          <div className="space-y-2">
            {Object.entries(mod.probabilities || {}).map(([cls, prob]) => (
              <div key={cls} className="flex items-center gap-2">
                <span className="text-gray-500 text-xs w-10 font-mono">{cls}</span>
                <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-brand-500 rounded-full"
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
                <span className="text-gray-400 text-xs w-12 text-right font-mono">
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>

          {isFake && (
            <div className="mt-4 grid grid-cols-2 gap-3 text-center text-sm">
              <div className="bg-gray-800 rounded-xl p-3">
                <p className="text-gray-500 text-xs mb-1">Audio Fake Prob.</p>
                <p className="text-amber-400 font-bold text-lg">
                  {((mod.audio_fake_prob || 0) * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-gray-800 rounded-xl p-3">
                <p className="text-gray-500 text-xs mb-1">Video Fake Prob.</p>
                <p className="text-orange-400 font-bold text-lg">
                  {((mod.video_fake_prob || 0) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Sync Timeline */}
      <div className="card">
        <h2 className="text-lg font-semibold text-white mb-1">Lip-Audio Sync Timeline</h2>
        <p className="text-gray-500 text-sm mb-4">
          Per-frame cosine similarity between lip and audio embeddings.
          Low values (below 0.5) indicate desynchronisation.
          {suspects.length > 0 && (
            <span className="text-red-400"> Red regions are suspect segments.</span>
          )}
        </p>
        {timeline.length > 0 ? (
          <SyncChart timeline={timeline} suspects={suspects} videoRef={videoRef} />
        ) : (
          <p className="text-gray-600 text-sm">No timeline data.</p>
        )}

        {suspects.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-gray-400 font-medium mb-2">Suspect Segments</p>
            <div className="flex flex-wrap gap-2">
              {suspects.map((seg, i) => (
                <span key={i} className="text-xs bg-red-500/10 border border-red-500/20 text-red-400 px-3 py-1 rounded-full font-mono">
                  {seg.start.toFixed(2)}s – {seg.end.toFixed(2)}s
                  <span className="text-red-600 ml-1">(sync={seg.mean_score.toFixed(3)})</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Evidence Frames */}
      {evidences.length > 0 && (
        <div className="card">
          <h2 className="text-lg font-semibold text-white mb-1">Evidence Frames</h2>
          <p className="text-gray-500 text-sm mb-4">
            Frames with lowest lip-audio sync scores (most suspicious moments).
            Timestamps are indicative — no frame-level ground truth exists.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {evidences.map((ef, i) => (
              <div key={i} className="bg-gray-800 rounded-xl overflow-hidden">
                <img
                  src={`data:image/jpeg;base64,${ef.frame_b64}`}
                  alt={`Evidence frame t=${ef.t}`}
                  className="w-full aspect-video object-cover"
                />
                <div className="flex gap-2 p-2">
                  <img
                    src={`data:image/jpeg;base64,${ef.mouth_b64}`}
                    alt="Mouth crop"
                    className="w-12 h-12 object-cover rounded-lg border border-gray-700"
                  />
                  <div>
                    <p className="text-white text-xs font-mono">{ef.t.toFixed(2)} s</p>
                    <p className="text-gray-400 text-xs">sync = {ef.score.toFixed(4)}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

