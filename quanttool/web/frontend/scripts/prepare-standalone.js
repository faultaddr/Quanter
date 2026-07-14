const { cpSync, existsSync, mkdirSync } = require('node:fs');
const path = require('node:path');

const frontendRoot = path.resolve(__dirname, '..');
const staticSource = path.join(frontendRoot, '.next/static');
const staticTarget = path.join(frontendRoot, '.next/standalone/.next/static');

if (!existsSync(staticSource)) {
  throw new Error('Next.js build assets are missing. Run npm run build first.');
}

mkdirSync(path.dirname(staticTarget), { recursive: true });
cpSync(staticSource, staticTarget, { recursive: true, force: true });

const publicSource = path.join(frontendRoot, 'public');
if (existsSync(publicSource)) {
  cpSync(publicSource, path.join(frontendRoot, '.next/standalone/public'), {
    recursive: true,
    force: true,
  });
}

console.log('Standalone assets prepared.');
