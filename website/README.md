# Website Template

This directory contains a medical clinic website template built with pure HTML, CSS, and JavaScript.

## Features

- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Modern UI**: Clean, professional design with smooth animations
- **No Dependencies**: Pure HTML/CSS/JS - no frameworks required
- **Accessible**: Includes ARIA labels and semantic HTML
- **Ready to Customize**: Easy to modify colors, content, and styling

## Structure

- `index.html` - Main website file (all-in-one with embedded CSS and JS)

## Getting Started

1. Open `index.html` in a web browser
2. Or serve it with a local web server:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Node.js (if you have http-server installed)
   npx http-server
   ```

Then visit `http://localhost:8000` in your browser.

## Customization

### Colors
Edit the CSS variables in the `:root` section:
- `--accent`: Primary brand color
- `--muted`: Secondary text color
- `--bg`: Background color
- `--card`: Card/container background

### Content
- Replace placeholder text with your clinic's information
- Update logo (currently an SVG data URI)
- Add real images to service cards
- Update team member information
- Modify contact information and location

### Forms
The contact and appointment forms currently show alerts. To make them functional:
1. Update `submitContact()` function to POST to your API endpoint
2. Update `submitAppt()` function to POST to your scheduling system

## Deployment

You can deploy this website to any static hosting service:
- GitHub Pages
- Netlify
- Vercel
- AWS S3 + CloudFront
- Any web server

Simply upload the `index.html` file (and any assets if you add them).


