/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./examples/**/*.html"],
  theme: {
    fontFamily: {
      'sans': ['Avenir'],
    },
    extend: {
      keyframes: {
        bounceX: {
          '0%, 100%': {
            left: '0',
            'animation-timing-function': 'cubic-bezier(0, 0, 0.2, 1)'
          },
          '50%': {
            left: '3px',
            'animation-timing-function': 'cubic-bezier(0.8, 0, 1, 1)'
          }
        }
      },
      animation: {
        bounceX: 'bounceX 1s infinite',
      }
    }
  },
  plugins: [],
}
