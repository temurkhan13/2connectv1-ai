const { chromium } = require('playwright');

const FRONTEND_URL = 'http://localhost:5173';
const BACKEND_URL = 'http://localhost:3000/api/v1';

// Generate unique emails with timestamp to avoid conflicts
const TS = Date.now();
const USERS = [
  { email: `test_alice_${TS}@test.com`, password: 'Test123!', firstName: 'Alice', lastName: 'Chen', role: 'Founder', objective: 'fundraising' },
  { email: `test_bob_${TS}@test.com`, password: 'Test123!', firstName: 'Bob', lastName: 'Smith', role: 'Investor', objective: 'investing' },
  { email: `test_carol_${TS}@test.com`, password: 'Test123!', firstName: 'Carol', lastName: 'Davis', role: 'Founder', objective: 'partnership' },
  { email: `test_david_${TS}@test.com`, password: 'Test123!', firstName: 'David', lastName: 'Lee', role: 'Executive', objective: 'hiring' },
  { email: `test_eva_${TS}@test.com`, password: 'Test123!', firstName: 'Eva', lastName: 'Martinez', role: 'Entrepreneur', objective: 'mentorship' },
];

// Helper to get verification code from database
async function getVerificationCode(email) {
  const { execSync } = require('child_process');
  try {
    const result = execSync(
      `docker exec reciprocity-postgres psql -U reciprocity_user -d reciprocity_ai -t -c "SELECT vc.code FROM verification_codes vc JOIN users u ON vc.user_id = u.id WHERE u.email = '${email}' AND vc.type = 'email_verification' ORDER BY vc.created_at DESC LIMIT 1;"`,
      { encoding: 'utf8' }
    );
    return result.trim();
  } catch (e) {
    console.log(`  [WARN] Could not get verification code: ${e.message.slice(0, 50)}`);
    return null;
  }
}

async function testUserJourney(browser, user, userNum) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`USER ${userNum}: ${user.firstName} ${user.lastName} (${user.role})`);
  console.log('='.repeat(60));

  const context = await browser.newContext();
  const page = await context.newPage();
  const results = { user: `${user.firstName} ${user.lastName}`, steps: {} };

  try {
    // Step 1: Visit home page
    console.log('\n[Step 1] Visiting home page...');
    await page.goto(FRONTEND_URL, { timeout: 10000 });
    const title = await page.title();
    console.log(`  [OK] Page loaded: ${title}`);
    results.steps['1_home'] = 'PASS';
    await page.screenshot({ path: `screenshots/user${userNum}_01_home.png` });

    // Step 2: Navigate to signup
    console.log('\n[Step 2] Going to signup...');
    await page.goto(`${FRONTEND_URL}/signup`, { timeout: 10000 });
    await page.waitForTimeout(1000);
    console.log(`  [OK] Navigated to signup, URL: ${page.url()}`);
    results.steps['2_signup_nav'] = 'PASS';
    await page.screenshot({ path: `screenshots/user${userNum}_02_signup.png` });

    // Step 3: Fill signup form
    console.log('\n[Step 3] Filling signup form...');
    try {
      await page.waitForTimeout(1000);

      // Fill first name
      const firstNameInput = await page.locator('input[placeholder*="First name" i]').first();
      if (await firstNameInput.isVisible()) {
        await firstNameInput.fill(user.firstName);
        console.log(`  [OK] Filled first name: ${user.firstName}`);
      }

      // Fill last name
      const lastNameInput = await page.locator('input[placeholder*="Last name" i]').first();
      if (await lastNameInput.isVisible()) {
        await lastNameInput.fill(user.lastName);
        console.log(`  [OK] Filled last name: ${user.lastName}`);
      }

      // Fill email
      const emailInput = await page.locator('input[type="email"], input[placeholder*="email" i]').first();
      if (await emailInput.isVisible()) {
        await emailInput.fill(user.email);
        console.log(`  [OK] Filled email: ${user.email}`);
      }

      // Fill password fields
      const passwordInputs = await page.locator('input[type="password"]').all();
      if (passwordInputs.length >= 1) {
        await passwordInputs[0].fill(user.password);
        console.log(`  [OK] Filled password`);
      }
      if (passwordInputs.length >= 2) {
        await passwordInputs[1].fill(user.password);
        console.log(`  [OK] Filled confirm password`);
      }

      results.steps['3_fill_form'] = 'PASS';
    } catch (e) {
      console.log(`  [FAIL] ${e.message.slice(0, 100)}`);
      results.steps['3_fill_form'] = 'FAIL';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_03_form.png` });

    // Step 4: Submit signup
    console.log('\n[Step 4] Submitting signup...');
    try {
      const submitBtn = await page.locator('button[type="submit"]:has-text("Create account")').first();
      if (await submitBtn.isVisible()) {
        await submitBtn.click();
        await page.waitForTimeout(3000);
        const url = page.url();
        console.log(`  [OK] Submitted, URL: ${url}`);
        results.steps['4_submit'] = url.includes('email-verification') ? 'PASS' : `AT: ${url.split('/').pop()}`;
      } else {
        console.log(`  [SKIP] No submit button found`);
        results.steps['4_submit'] = 'SKIP';
      }
    } catch (e) {
      console.log(`  [FAIL] ${e.message.slice(0, 100)}`);
      results.steps['4_submit'] = 'FAIL';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_04_submit.png` });

    // Step 5: Email Verification
    console.log('\n[Step 5] Email verification...');
    try {
      if (page.url().includes('email-verification')) {
        // Get verification code from database
        await page.waitForTimeout(1000);
        const code = await getVerificationCode(user.email);

        if (code) {
          console.log(`  [INFO] Got verification code: ${code}`);

          // Find OTP input fields and fill them
          const otpInputs = await page.locator('input[maxlength="1"], input.otp-input').all();
          if (otpInputs.length >= 4) {
            for (let i = 0; i < Math.min(code.length, otpInputs.length); i++) {
              await otpInputs[i].fill(code[i]);
            }
            console.log(`  [OK] Filled OTP code`);

            // Click verify button
            const verifyBtn = await page.locator('button[type="submit"], button:has-text("Verify")').first();
            if (await verifyBtn.isVisible()) {
              await verifyBtn.click();
              await page.waitForTimeout(3000);
              const url = page.url();
              console.log(`  [OK] Verified, URL: ${url}`);
              results.steps['5_email_verify'] = url.includes('onboarding') ? 'PASS' : `AT: ${url.split('/').pop()}`;
            }
          } else {
            console.log(`  [INFO] Found ${otpInputs.length} OTP inputs, trying single input`);
            const singleInput = await page.locator('input[name="code"], input[placeholder*="code" i]').first();
            if (await singleInput.isVisible()) {
              await singleInput.fill(code);
              const verifyBtn = await page.locator('button[type="submit"]').first();
              if (await verifyBtn.isVisible()) {
                await verifyBtn.click();
                await page.waitForTimeout(3000);
                results.steps['5_email_verify'] = 'PASS (single input)';
              }
            } else {
              results.steps['5_email_verify'] = 'SKIP (no input found)';
            }
          }
        } else {
          results.steps['5_email_verify'] = 'SKIP (no code)';
        }
      } else {
        results.steps['5_email_verify'] = `SKIP (not at verification: ${page.url().split('/').pop()})`;
      }
    } catch (e) {
      console.log(`  [FAIL] ${e.message.slice(0, 100)}`);
      results.steps['5_email_verify'] = 'FAIL';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_05_verify.png` });

    // Step 6: Onboarding
    console.log('\n[Step 6] Onboarding...');
    try {
      if (!page.url().includes('onboarding')) {
        await page.goto(`${FRONTEND_URL}/onboarding`, { timeout: 10000 });
        await page.waitForTimeout(2000);
      }

      const url = page.url();
      if (url.includes('onboarding')) {
        console.log(`  [OK] At onboarding: ${url}`);

        // Check for chat interface
        const chatInput = await page.locator('textarea, input[type="text"]:not([type="email"])').first();
        if (await chatInput.isVisible().catch(() => false)) {
          console.log(`  [OK] Chat input found`);
          results.steps['6_onboarding'] = 'PASS (chat visible)';
        } else {
          results.steps['6_onboarding'] = 'PASS';
        }
      } else {
        console.log(`  [INFO] Redirected to: ${url}`);
        results.steps['6_onboarding'] = `REDIRECT: ${url.split('/').pop()}`;
      }
    } catch (e) {
      console.log(`  [FAIL] ${e.message.slice(0, 100)}`);
      results.steps['6_onboarding'] = 'FAIL';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_06_onboarding.png` });

    // Step 7: Dashboard
    console.log('\n[Step 7] Dashboard...');
    try {
      await page.goto(`${FRONTEND_URL}/dashboard`, { timeout: 10000 });
      await page.waitForTimeout(2000);
      const dashUrl = page.url();
      console.log(`  [OK] Dashboard attempt, URL: ${dashUrl}`);
      results.steps['7_dashboard'] = dashUrl.includes('dashboard') ? 'PASS' : `REDIRECT: ${dashUrl.split('/').pop()}`;
    } catch (e) {
      console.log(`  [FAIL] ${e.message.slice(0, 100)}`);
      results.steps['7_dashboard'] = 'FAIL';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_07_dashboard.png` });

    // Step 8: Discover
    console.log('\n[Step 8] Discover...');
    try {
      await page.goto(`${FRONTEND_URL}/discover`, { timeout: 10000 });
      await page.waitForTimeout(2000);
      const discoverUrl = page.url();
      console.log(`  [OK] Discover attempt, URL: ${discoverUrl}`);
      results.steps['8_discover'] = discoverUrl.includes('discover') ? 'PASS' : `REDIRECT: ${discoverUrl.split('/').pop()}`;
    } catch (e) {
      console.log(`  [FAIL] ${e.message.slice(0, 100)}`);
      results.steps['8_discover'] = 'FAIL';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_08_discover.png` });

  } catch (error) {
    console.log(`  [ERROR] ${error.message}`);
    results.steps['ERROR'] = error.message.slice(0, 100);
  } finally {
    await context.close();
  }

  return results;
}

async function main() {
  console.log('='.repeat(60));
  console.log('RECIPROCITY PLATFORM - BROWSER E2E TEST');
  console.log(`Started: ${new Date().toISOString()}`);
  console.log('='.repeat(60));

  // Create screenshots directory
  const fs = require('fs');
  if (!fs.existsSync('screenshots')) {
    fs.mkdirSync('screenshots');
  }

  const browser = await chromium.launch({ headless: true });
  const allResults = [];

  for (let i = 0; i < USERS.length; i++) {
    const result = await testUserJourney(browser, USERS[i], i + 1);
    allResults.push(result);
  }

  await browser.close();

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));

  let totalPassed = 0;
  let totalSteps = 0;

  for (const result of allResults) {
    const steps = result.steps;
    const passed = Object.values(steps).filter(s => String(s).includes('PASS')).length;
    const total = Object.keys(steps).length;
    totalPassed += passed;
    totalSteps += total;

    console.log(`\n${result.user}: ${passed}/${total} steps passed`);
    for (const [step, status] of Object.entries(steps)) {
      const icon = String(status).includes('PASS') ? '[OK]' : '[X]';
      console.log(`  ${icon} ${step}: ${status}`);
    }
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log(`OVERALL: ${totalPassed}/${totalSteps} steps passed (${Math.round(100 * totalPassed / totalSteps)}%)`);
  console.log(`Screenshots saved to: screenshots/`);
  console.log(`Completed: ${new Date().toISOString()}`);
}

main().catch(console.error);
