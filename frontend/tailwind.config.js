/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#e8eaf6',
          100: '#c5cae9',
          500: '#3949ab',
          600: '#1565c0',
          700: '#0d47a1',
        },
        danger: {
          50:  '#ffebee',
          500: '#c62828',
          600: '#b71c1c',
        },
        success: {
          50:  '#e8f5e9',
          500: '#2e7d32',
          600: '#1b5e20',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}

