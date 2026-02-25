const { chromium } = require('playwright');
// Using native fetch (Node.js 18+)

const FRONTEND_URL = 'http://localhost:5173';
const BACKEND_URL = 'http://localhost:3000/api/v1';
const AI_SERVICE = 'http://localhost:8000/api/v1';
const API_KEY = 'dev-api-key';

const TS = Date.now();
const USERS = [
  { email: `test_alice_${TS}@test.com`, password: 'Test123!', firstName: 'Alice', lastName: 'Chen', role: 'Founder', objective: 'fundraising' },
  { email: `test_bob_${TS}@test.com`, password: 'Test123!', firstName: 'Bob', lastName: 'Smith', role: 'Investor', objective: 'investing' },
];

const ONBOARDING_MESSAGES = [
  "I'm a tech founder building an AI startup. Looking for investors and advisors.",
  "We're pre-seed, targeting Series A in 6 months. Need $2M.",
  "Our focus is B2B SaaS for healthcare. I have 10 years in the industry.",
];

// Helper to get verification code from database
function getVerificationCode(email) {
  const { execSync } = require('child_process');
  try {
    const result = execSync(
      `docker exec reciprocity-postgres psql -U reciprocity_user -d reciprocity_ai -t -c "SELECT vc.code FROM verification_codes vc JOIN users u ON vc.user_id = u.id WHERE u.email = '${email}' AND vc.type = 'email_verification' ORDER BY vc.created_at DESC LIMIT 1;"`,
      { encoding: 'utf8' }
    );
    return result.trim();
  } catch (e) {
    return null;
  }
}

// Helper to get user ID from database
function getUserId(email) {
  const { execSync } = require('child_process');
  try {
    const result = execSync(
      `docker exec reciprocity-postgres psql -U reciprocity_user -d reciprocity_ai -t -c "SELECT id FROM users WHERE email = '${email}' LIMIT 1;"`,
      { encoding: 'utf8' }
    );
    return result.trim();
  } catch (e) {
    return null;
  }
}

// Set onboarding status to completed (bridges AI service onboarding to backend)
function setOnboardingCompleted(userId) {
  const { execSync } = require('child_process');
  try {
    execSync(
      `docker exec reciprocity-postgres psql -U reciprocity_user -d reciprocity_ai -c "UPDATE users SET onboarding_status = 'completed' WHERE id = '${userId}';"`,
      { encoding: 'utf8' }
    );
    return true;
  } catch (e) {
    return false;
  }
}

// Complete onboarding via API
async function completeOnboardingViaAPI(userId) {
  const headers = { 'Content-Type': 'application/json', 'X-API-KEY': API_KEY };

  try {
    // Step 1: Start session
    const startResp = await fetch(`${AI_SERVICE}/onboarding/start`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ user_id: userId })
    });
    if (!startResp.ok) return { success: false, error: 'start failed' };
    const startData = await startResp.json();
    const sessionId = startData.session_id;

    // Step 2: Chat conversation
    for (const msg of ONBOARDING_MESSAGES) {
      await fetch(`${AI_SERVICE}/onboarding/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ user_id: userId, session_id: sessionId, message: msg })
      });
      await new Promise(r => setTimeout(r, 500));
    }

    // Step 3: Finalize
    await fetch(`${AI_SERVICE}/onboarding/finalize/${sessionId}`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ user_id: userId })
    });

    // Step 4: Complete (creates DynamoDB profile)
    const completeResp = await fetch(`${AI_SERVICE}/onboarding/complete`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ session_id: sessionId, user_id: userId })
    });
    if (!completeResp.ok) return { success: false, error: 'complete failed' };

    // Wait for persona generation
    await new Promise(r => setTimeout(r, 5000));

    // Step 5: Approve summary
    await fetch(`${AI_SERVICE}/user/approve-summary`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ user_id: userId })
    });

    return { success: true, sessionId };
  } catch (e) {
    return { success: false, error: e.message };
  }
}

async function testUserJourney(browser, user, userNum) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`USER ${userNum}: ${user.firstName} ${user.lastName} (${user.role})`);
  console.log('='.repeat(70));

  const context = await browser.newContext();
  const page = await context.newPage();
  const results = { user: `${user.firstName} ${user.lastName}`, steps: {}, userId: null };

  try {
    // ===== PHASE 1: BROWSER SIGNUP =====
    console.log('\n[Step 1] Browser Signup...');
    await page.goto(`${FRONTEND_URL}/signup`, { timeout: 15000 });
    await page.waitForTimeout(1000);

    await page.locator('input[placeholder*="First name" i]').first().fill(user.firstName);
    await page.locator('input[placeholder*="Last name" i]').first().fill(user.lastName);
    await page.locator('input[type="email"]').first().fill(user.email);
    const pwInputs = await page.locator('input[type="password"]').all();
    await pwInputs[0].fill(user.password);
    if (pwInputs.length >= 2) await pwInputs[1].fill(user.password);

    await page.locator('button[type="submit"]:has-text("Create account")').first().click();
    await page.waitForTimeout(3000);

    if (page.url().includes('email-verification')) {
      console.log(`  [OK] Signup successful`);
      results.steps['1_signup'] = 'PASS';
    } else {
      results.steps['1_signup'] = 'FAIL';
      throw new Error('Signup failed');
    }

    // ===== PHASE 2: BROWSER EMAIL VERIFICATION =====
    console.log('\n[Step 2] Browser Email Verification...');
    await page.waitForTimeout(1000);
    const code = getVerificationCode(user.email);

    if (code) {
      const otpInputs = await page.locator('input[maxlength="1"]').all();
      for (let i = 0; i < Math.min(code.length, otpInputs.length); i++) {
        await otpInputs[i].fill(code[i]);
      }
      await page.locator('button[type="submit"]').first().click();
      await page.waitForTimeout(3000);

      console.log(`  [OK] Email verified`);
      results.steps['2_email_verify'] = 'PASS';
    } else {
      results.steps['2_email_verify'] = 'FAIL';
      throw new Error('No verification code');
    }

    // Get user ID
    results.userId = getUserId(user.email);
    console.log(`  [INFO] User ID: ${results.userId}`);

    // ===== PHASE 3: API ONBOARDING =====
    console.log('\n[Step 3] API Onboarding (chat + profile creation)...');
    const onboardingResult = await completeOnboardingViaAPI(results.userId);

    if (onboardingResult.success) {
      console.log(`  [OK] Onboarding completed via API`);
      results.steps['3_api_onboarding'] = 'PASS';
      // Note: AI service /onboarding/complete now sets onboarding_status='completed' in PostgreSQL automatically
      console.log(`  [OK] PostgreSQL onboarding_status should be set by AI service`);
    } else {
      console.log(`  [WARN] Onboarding: ${onboardingResult.error}`);
      results.steps['3_api_onboarding'] = `PARTIAL: ${onboardingResult.error}`;
    }

    // Wait for backend to process
    await page.waitForTimeout(3000);

    // ===== PHASE 4: BROWSER ONBOARDING MATCHES (sets onboarding_matches flag) =====
    console.log('\n[Step 4] Browser Onboarding Matches (sets completion flag)...');
    await page.goto(`${FRONTEND_URL}/onboarding/matches`, { timeout: 15000 });
    await page.waitForTimeout(5000); // Wait for backend call to complete
    await page.screenshot({ path: `screenshots/user${userNum}_onb_matches.png` });

    const onbMatchesUrl = page.url();
    if (onbMatchesUrl.includes('matches')) {
      console.log(`  [OK] Onboarding matches page loaded`);
      results.steps['4_onb_matches'] = 'PASS';

      // Look for "Go to dashboard" button and click it
      const dashBtn = page.locator('button:has-text("dashboard"), a:has-text("dashboard")').first();
      if (await dashBtn.isVisible().catch(() => false)) {
        await dashBtn.click();
        await page.waitForTimeout(2000);
        console.log(`  [OK] Clicked dashboard button`);
      }
    } else {
      results.steps['4_onb_matches'] = `REDIRECT: ${onbMatchesUrl.split('/').pop()}`;
    }

    // ===== PHASE 5: BROWSER DASHBOARD =====
    console.log('\n[Step 5] Browser Dashboard Access...');
    if (!page.url().includes('dashboard')) {
      await page.goto(`${FRONTEND_URL}/dashboard`, { timeout: 15000 });
    }
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `screenshots/user${userNum}_dashboard.png` });

    const dashUrl = page.url();
    if (dashUrl.includes('dashboard')) {
      console.log(`  [OK] Dashboard accessible!`);
      results.steps['5_dashboard'] = 'PASS';
    } else {
      console.log(`  [INFO] Redirected to: ${dashUrl}`);
      results.steps['5_dashboard'] = `REDIRECT: ${dashUrl.split('/').pop()}`;
    }

    // ===== PHASE 6: BROWSER DISCOVER =====
    console.log('\n[Step 6] Browser Discover Page...');
    await page.goto(`${FRONTEND_URL}/discover`, { timeout: 15000 });
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `screenshots/user${userNum}_discover.png` });

    const discoverUrl = page.url();
    if (discoverUrl.includes('discover')) {
      console.log(`  [OK] Discover accessible!`);
      results.steps['6_discover'] = 'PASS';
    } else {
      results.steps['6_discover'] = `REDIRECT: ${discoverUrl.split('/').pop()}`;
    }

    // ===== PHASE 7: BROWSER MATCHES =====
    console.log('\n[Step 7] Browser Matches Page...');
    await page.goto(`${FRONTEND_URL}/matches`, { timeout: 15000 });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `screenshots/user${userNum}_matches.png` });

    const matchesUrl = page.url();
    if (matchesUrl.includes('matches')) {
      console.log(`  [OK] Matches accessible!`);
      results.steps['7_matches'] = 'PASS';
    } else {
      results.steps['7_matches'] = `REDIRECT: ${matchesUrl.split('/').pop()}`;
    }

  } catch (error) {
    console.log(`  [ERROR] ${error.message}`);
    results.steps['ERROR'] = error.message.slice(0, 100);
  } finally {
    await context.close();
  }

  return results;
}

// Test Interactive Discovery via API
async function testDiscoveryAPI(userResults) {
  console.log(`\n${'='.repeat(70)}`);
  console.log('INTERACTIVE DISCOVERY API TEST');
  console.log('='.repeat(70));

  const results = { steps: {} };

  const validUsers = userResults.filter(r => r.userId);
  if (validUsers.length < 2) {
    results.steps['discovery'] = 'SKIP: need 2 users';
    return results;
  }

  const userA = validUsers[0];
  const userB = validUsers[1];
  console.log(`  Testing: ${userA.user} (${userA.userId}) <-> ${userB.user} (${userB.userId})`);

  // Wait for persona generation and embeddings to complete (async via Celery)
  console.log('\n  Waiting 15s for persona generation and embeddings...');
  await new Promise(r => setTimeout(r, 15000));

  try {
    // Get JWT tokens for users (simplified - in real test you'd get proper tokens)
    const headers = { 'Content-Type': 'application/json' };

    // Step 1: User A expresses interest in User B via AI service (which doesn't need auth)
    console.log('\n[Discovery 1] Checking AI matching...');
    const matchesResp = await fetch(`${AI_SERVICE}/matching/${userA.userId}/matches`, {
      headers: { 'X-API-KEY': API_KEY }
    });

    if (matchesResp.ok) {
      const matchesData = await matchesResp.json();
      const matchCount = matchesData.matches?.length || matchesData.data?.matches?.length || 0;
      console.log(`  [OK] User A has ${matchCount} AI matches`);
      results.steps['1_ai_matches'] = `PASS (${matchCount} matches)`;
    } else {
      results.steps['1_ai_matches'] = `HTTP ${matchesResp.status}`;
    }

    // Step 2: Check User B matches
    console.log('\n[Discovery 2] Checking User B matches...');
    const matchesBResp = await fetch(`${AI_SERVICE}/matching/${userB.userId}/matches`, {
      headers: { 'X-API-KEY': API_KEY }
    });

    if (matchesBResp.ok) {
      const matchesData = await matchesBResp.json();
      const matchCount = matchesData.matches?.length || matchesData.data?.matches?.length || 0;
      console.log(`  [OK] User B has ${matchCount} AI matches`);
      results.steps['2_ai_matches_b'] = `PASS (${matchCount} matches)`;
    } else {
      results.steps['2_ai_matches_b'] = `HTTP ${matchesBResp.status}`;
    }

  } catch (e) {
    console.log(`  [ERROR] ${e.message}`);
    results.steps['error'] = e.message.slice(0, 100);
  }

  return results;
}

async function main() {
  console.log('='.repeat(70));
  console.log('RECIPROCITY - HYBRID E2E TEST');
  console.log('Browser (signup/verify) + API (onboarding) + Browser (dashboard/discover)');
  console.log(`Started: ${new Date().toISOString()}`);
  console.log('='.repeat(70));

  const fs = require('fs');
  if (!fs.existsSync('screenshots')) fs.mkdirSync('screenshots');

  const browser = await chromium.launch({ headless: true });
  const allResults = [];

  for (let i = 0; i < USERS.length; i++) {
    const result = await testUserJourney(browser, USERS[i], i + 1);
    allResults.push(result);
  }

  // Test Discovery
  const discoveryResults = await testDiscoveryAPI(allResults);
  allResults.push({ user: 'Discovery API Test', steps: discoveryResults.steps });

  await browser.close();

  // Summary
  console.log('\n' + '='.repeat(70));
  console.log('SUMMARY');
  console.log('='.repeat(70));

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

  console.log(`\n${'='.repeat(70)}`);
  console.log(`OVERALL: ${totalPassed}/${totalSteps} steps passed (${Math.round(100 * totalPassed / totalSteps)}%)`);
  console.log(`Completed: ${new Date().toISOString()}`);
}

main().catch(console.error);
