# Deployment Guide

## What Was Fixed

The deployment error `npm error enoent Could not read package.json` occurred because:
1. Missing root `package.json` — deployment platform looked for it but found none
2. Build configuration didn't specify how to build the frontend
3. No `vercel.json` to guide the deployment platform

## Files Created

### Root Configuration
- **`package.json`** — Root-level build scripts that run `npm install` and `npm run build` in the frontend directory
- **`vercel.json`** — Deployment configuration for Vercel (specifies build command, output directory)
- **`.nvmrc`** — Node version lock (18) for consistency

## How to Deploy

### Option 1: Vercel (Recommended for Frontend)

1. Push to GitHub:
   ```bash
   git add .
   git commit -m "Deploy DocReader AI"
   git push
   ```

2. Go to [vercel.com](https://vercel.com) and import your GitHub repo

3. **Build Settings:**
   - Build Command: `npm run build` (uses root package.json)
   - Output Directory: `frontend/dist`
   - Framework Preset: Other / Vite

4. **Environment Variables:**
   - Add `VITE_API_BASE_URL` pointing to your FastAPI backend

5. Deploy! Vercel automatically runs the build, which:
   - Installs frontend dependencies
   - Builds React app with Vite
   - Outputs static files to `frontend/dist/`
   - Serves via Vercel CDN

### Option 2: Self-Hosted (Any Static Host)

```bash
# Build locally
npm run build

# Upload frontend/dist/ to your hosting:
# - Netlify
# - GitHub Pages
# - AWS S3 + CloudFront
# - Any static web server
```

### Backend Deployment

Deploy the FastAPI backend separately to:
- **Render**: Connect repo, select Python, set start command to `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Railway**: Similar setup
- **Heroku**: Legacy but still works
- **DigitalOcean App Platform**: App spec YAML
- **Self-hosted**: Any Linux server with Python 3.8+

Then set `VITE_API_BASE_URL` environment variable in your frontend deployment to point to the backend URL.

## Build Flow

```
Push to GitHub
       ↓
Vercel detects changes
       ↓
npm run build (from root package.json)
       ↓
cd frontend && npm install && npm run build
       ↓
Vite compiles React app
       ↓
Output: frontend/dist/
       ↓
Vercel serves static files
       ↓
Live at your domain
```

## Troubleshooting

### Build Fails with "vite: not found"
- Ensure Node 18+ is installed
- Vercel should auto-detect from `.nvmrc`
- Check `package.json` scripts are correct

### API Calls Fail After Deploy
- Backend must be running and accessible
- `VITE_API_BASE_URL` must be set to backend domain
- Check CORS is enabled in FastAPI (`CORSMiddleware`)
- Verify firewall/security groups allow traffic

### Speech Not Working
- SpeechSynthesis API requires HTTPS (except localhost)
- Ensure your Vercel domain uses HTTPS
- Test in browser console: `window.speechSynthesis ? 'OK' : 'Fail'`

## Environment Variables Reference

### Frontend (.env in Vercel)
```
VITE_API_BASE_URL=https://api.yourdomain.com
```

### Backend
```
CHART_CAPTIONER_CHECKPOINT=checkpoints/best.pt  (optional)
PORT=8000
```

## Next Steps

1. **Verify build works locally:**
   ```bash
   npm run build
   npm run preview
   ```

2. **Deploy frontend to Vercel**

3. **Deploy backend to Render or similar**

4. **Set API URL in Vercel env vars**

5. **Test in production** — visit your domain, upload a test PDF, verify speech works

---

See README.md for full setup and development instructions.
