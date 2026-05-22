/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        glow: "0 0 0 1px rgba(148, 163, 184, 0.12), 0 24px 80px rgba(2, 6, 23, 0.28)",
      },
      fontFamily: {
        display: ['"Space Grotesk"', "sans-serif"],
      },
    },
  },
  plugins: [],
};
