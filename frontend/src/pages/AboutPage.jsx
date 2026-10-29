import { AlertTriangle, BookOpen, Database, Cpu, Globe } from 'lucide-react'

function Section({ icon: Icon, title, children }) {
  return (
    <section className="card">
      <div className="flex items-center gap-3 mb-4">
        <Icon className="w-6 h-6 text-brand-500" />
        <h2 className="text-xl font-semibold text-white">{title}</h2>
      </div>
      <div className="text-gray-400 text-sm leading-7 space-y-2">{children}</div>
    </section>
  )
}

export default function AboutPage() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-12 space-y-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-white">About AV-Forensics</h1>
        <p className="text-gray-400 mt-2">
          Research system for audio-visual deepfake detection via lip-sync consistency.
        </p>
      </div>

      <Section icon={Cpu} title="How It Works">
        <p>
          The system analyses uploaded videos by checking whether the speaker's lip movements
          are consistent with the audio track — a property that AI-generated deepfakes often violate.
        </p>
        <ol className="list-decimal list-inside space-y-1 mt-2">
          <li>The video is split into 4-second windows (2-second stride).</li>
          <li>Mouth-region frames are extracted with MediaPipe face mesh and encoded by a
            <strong> Lip Video Transformer</strong> (spatial + temporal attention, 256-dim).</li>
          <li>Audio is resampled to 16 kHz and encoded by <strong>mHuBERT-147</strong>
            (learnable weighted sum of hidden layers).</li>
          <li>A <strong>cross-modal transformer</strong> fuses visual and audio tokens.
            Cosine similarity between pre-fusion embeddings forms the sync timeline.</li>
          <li>Two MLP heads produce: (a) Real / Fake verdict, (b) 4-class modality prediction
            (RVRA / RVFA / FVRA / FVFA).</li>
          <li>Window predictions are averaged; the sync timeline highlights suspicious frames.</li>
        </ol>
      </Section>

      <Section icon={Database} title="Training Dataset">
        <p>
          The model is trained exclusively on{' '}
          <a href="https://www.kaggle.com/datasets/shreyaty08/fakeavceleb" target="_blank" rel="noopener noreferrer"
            className="text-brand-500 underline hover:text-brand-300">FakeAVCeleb</a>{' '}
          (Kaggle: shreyaty08/fakeavceleb), which contains four categories:
        </p>
        <ul className="list-disc list-inside space-y-0.5 mt-2">
          <li><strong className="text-white">RVRA</strong> — Real Video + Real Audio → Real</li>
          <li><strong className="text-white">RVFA</strong> — Real Video + Fake Audio → Fake</li>
          <li><strong className="text-white">FVRA</strong> — Fake Video + Real Audio → Fake</li>
          <li><strong className="text-white">FVFA</strong> — Fake Video + Fake Audio → Fake</li>
        </ul>
        <p className="mt-2">
          Identity-grouped train/val splits prevent speaker leakage. Class weights
          compensate for the smaller RVRA set.
        </p>
      </Section>

      <Section icon={BookOpen} title="Methodology Details">
        <p>
          A sync consistency loss is applied <em>only</em> to real (RVRA) pairs:
          it maximises cosine similarity between aligned lip and audio embeddings,
          with time-shifted audio as contrastive negatives. Fake pairs are never
          pulled together — the model learns genuine sync without overfitting to
          fake distributions.
        </p>
        <p>
          Total training loss = CE(binary) + CE(4-class) + 0.5 × (CE audio-fake + CE video-fake)
          + 0.3 × sync-consistency loss.
        </p>
      </Section>

      <Section icon={Globe} title="Deployment">
        <p>
          <strong className="text-white">Frontend:</strong> React + Vite + Tailwind CSS,
          deployed on Vercel / Netlify (static, free tier).
        </p>
        <p>
          <strong className="text-white">Backend:</strong> FastAPI + PyTorch (CPU inference),
          deployed as a Docker container on Hugging Face Spaces (free tier, may cold-start in ~30 s).
        </p>
        <p>
          Uploaded videos are deleted immediately after processing. Results are held in
          memory for approximately one hour, then purged.
        </p>
      </Section>

      <div className="card border-amber-600/30 bg-amber-600/5">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-6 h-6 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <h2 className="text-lg font-semibold text-amber-300 mb-3">Limitations</h2>
            <ul className="text-gray-400 text-sm leading-7 space-y-1 list-disc list-inside">
              <li>Results are <strong className="text-white">probabilistic</strong>, not certainties.</li>
              <li>Trained only on <strong className="text-white">FakeAVCeleb</strong> (English speech, celebrity videos). Accuracy on other languages, accents, or conditions has not been validated.</li>
              <li>Multilingual deepfakes, non-speech audio, or heavily compressed video may degrade accuracy.</li>
              <li>Evidence timestamps are <strong className="text-white">indicative only</strong> — training data has no frame-level labels.</li>
              <li>This tool is <strong className="text-white">not legal proof</strong> of any kind. Do not use it as the sole basis for legal, financial, or editorial decisions.</li>
              <li>The system does not detect all types of deepfakes — only audio-visual desynchronisation is modelled.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

