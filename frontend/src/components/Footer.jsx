export default function Footer() {
  return (
    <footer className="border-t border-gray-800 py-6 text-center text-xs text-gray-500">
      <p>
        AV-Forensics — Audio-Visual Deepfake Detection ·{' '}
        Trained on{' '}
        <a
          href="https://www.kaggle.com/datasets/shreyaty08/fakeavceleb"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-gray-300"
        >
          FakeAVCeleb
        </a>
      </p>
      <p className="mt-1 text-gray-600">
        Results are probabilistic and NOT legal proof. See About for limitations.
      </p>
    </footer>
  )
}

