const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({ args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 2 });
  const html = path.resolve(__dirname, 'architecture.html');
  await page.goto(`file://${html}`, { waitUntil: 'networkidle0' });
  // Let fonts settle
  await new Promise(r => setTimeout(r, 400));
  // Full page height
  const height = await page.evaluate(() => document.body.scrollHeight);
  await page.setViewport({ width: 1400, height, deviceScaleFactor: 2 });
  await page.screenshot({ path: path.resolve(__dirname, 'architecture.png'), fullPage: true });
  await browser.close();
  console.log('✅  architecture.png saved');
})();
