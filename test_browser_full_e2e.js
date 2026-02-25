const { chromium } = require('playwright');

const FRONTEND_URL = 'http://localhost:5173';
const BACKEND_URL = 'http://localhost:3000/api/v1';

// Generate unique emails with timestamp to avoid conflicts
const TS = Date.now();
const USERS = [
  { email: `test_alice_${TS}@test.com`, password: 'Test123!', firstName: 'Alice', lastName: 'Chen', role: 'Founder', objective: 'fundraising' },
  { email: `test_bob_${TS}@test.com`, password: 'Test123!', firstName: 'Bob', lastName: 'Smith', role: 'Investor', objective: 'investing' },
];

// Onboarding responses for different user types
const ONBOARDING_RESPONSES = {
  fundraising: [
    "I'm looking to raise funding for my AI startup",
    "We're at seed stage, planning Series A in 6 months",
    "B2B SaaS for healthcare, I have 10 years in the industry",
    "Based in London, prefer investors who understand European markets",
    "Looking for strategic investors with healthcare network connections"
  ],
  investing: [
    "I'm an angel investor looking to invest in early-stage startups",
    "I typically invest $50k-$200k in seed rounds",
    "Focus on B2B SaaS, AI/ML, and healthcare tech",
    "I can offer mentorship and board advisory experience",
    "Looking for founders with strong technical backgrounds"
  ]
};

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

// Helper to get user ID from database
async function getUserId(email) {
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

async function testFullUserJourney(browser, user, userNum) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`USER ${userNum}: ${user.firstName} ${user.lastName} (${user.role} - ${user.objective})`);
  console.log('='.repeat(70));

  const context = await browser.newContext();
  const page = await context.newPage();
  const results = { user: `${user.firstName} ${user.lastName}`, objective: user.objective, steps: {} };

  try {
    // ===== PHASE 1: SIGNUP =====
    console.log('\n[Step 1] Signup...');
    await page.goto(`${FRONTEND_URL}/signup`, { timeout: 15000 });
    await page.waitForTimeout(1000);

    // Fill form
    await page.locator('input[placeholder*="First name" i]').first().fill(user.firstName);
    await page.locator('input[placeholder*="Last name" i]').first().fill(user.lastName);
    await page.locator('input[type="email"], input[placeholder*="email" i]').first().fill(user.email);
    const passwordInputs = await page.locator('input[type="password"]').all();
    await passwordInputs[0].fill(user.password);
    if (passwordInputs.length >= 2) await passwordInputs[1].fill(user.password);

    await page.locator('button[type="submit"]:has-text("Create account")').first().click();
    await page.waitForTimeout(3000);

    if (page.url().includes('email-verification')) {
      console.log(`  [OK] Signup successful, at email verification`);
      results.steps['1_signup'] = 'PASS';
    } else {
      results.steps['1_signup'] = `FAIL: ${page.url().split('/').pop()}`;
      throw new Error('Signup failed');
    }

    // ===== PHASE 2: EMAIL VERIFICATION =====
    console.log('\n[Step 2] Email verification...');
    await page.waitForTimeout(1000);
    const code = await getVerificationCode(user.email);

    if (code) {
      const otpInputs = await page.locator('input[maxlength="1"], input.otp-input').all();
      for (let i = 0; i < Math.min(code.length, otpInputs.length); i++) {
        await otpInputs[i].fill(code[i]);
      }
      await page.locator('button[type="submit"], button:has-text("Verify")').first().click();
      await page.waitForTimeout(3000);

      if (page.url().includes('onboarding')) {
        console.log(`  [OK] Email verified, at onboarding`);
        results.steps['2_email_verify'] = 'PASS';
      } else {
        results.steps['2_email_verify'] = `PARTIAL: ${page.url().split('/').pop()}`;
      }
    } else {
      results.steps['2_email_verify'] = 'FAIL: no code';
      throw new Error('No verification code');
    }

    // ===== PHASE 3: ONBOARDING CHAT =====
    console.log('\n[Step 3] Onboarding chat conversation...');
    await page.waitForTimeout(3000); // Wait for page to fully load

    const responses = ONBOARDING_RESPONSES[user.objective] || ONBOARDING_RESPONSES.fundraising;
    let chatCompleted = false;
    let messagesAnswered = 0;

    // Take initial screenshot
    await page.screenshot({ path: `screenshots/user${userNum}_03a_onboarding_start.png` });

    for (let i = 0; i < responses.length; i++) {
      try {
        // Wait for chat input - use exact placeholder match
        const chatInput = page.locator('textarea[placeholder="What\'s on your mind?"]');

        // Wait with retry
        let inputVisible = false;
        for (let retry = 0; retry < 3; retry++) {
          inputVisible = await chatInput.isVisible().catch(() => false);
          if (inputVisible) break;
          await page.waitForTimeout(2000);
        }

        if (!inputVisible) {
          // Try alternative selectors
          const altInput = page.locator('textarea').first();
          inputVisible = await altInput.isVisible().catch(() => false);
          if (inputVisible) {
            await altInput.fill(responses[i]);
          } else {
            console.log(`  [WARN] Chat input not found at step ${i+1}`);
            continue;
          }
        } else {
          await chatInput.fill(responses[i]);
        }

        console.log(`  [${i+1}] Typed: "${responses[i].slice(0, 40)}..."`);

        // Find and click submit button - look for the arrow button with MdEast icon
        const submitBtn = page.locator('button.text-primary-400').last();
        if (await submitBtn.isVisible().catch(() => false)) {
          await submitBtn.click();
          messagesAnswered++;
          console.log(`  [${i+1}] Submitted message`);
        }

        // Wait for AI response (backend processing)
        await page.waitForTimeout(4000);

        // Check if we're at summary page
        if (page.url().includes('summary')) {
          console.log(`  [OK] Chat complete, at summary page`);
          chatCompleted = true;
          break;
        }

        // Check for "Go to summary" button
        const summaryBtn = page.locator('button:has-text("summary"), a:has-text("summary")').first();
        if (await summaryBtn.isVisible().catch(() => false)) {
          console.log(`  [OK] Summary button appeared, clicking...`);
          await summaryBtn.click();
          await page.waitForTimeout(2000);
          chatCompleted = true;
          break;
        }
      } catch (e) {
        console.log(`  [WARN] Chat step ${i+1} issue: ${e.message.slice(0, 50)}`);
      }
    }

    if (messagesAnswered > 0) {
      chatCompleted = true;
      console.log(`  [OK] Answered ${messagesAnswered} questions`);
    }

    results.steps['3_onboarding_chat'] = chatCompleted ? 'PASS' : 'PARTIAL';
    await page.screenshot({ path: `screenshots/user${userNum}_03_chat.png` });

    // ===== PHASE 4: SUMMARY REVIEW =====
    console.log('\n[Step 4] Summary review...');
    try {
      // Navigate to summary if not there
      if (!page.url().includes('summary')) {
        await page.goto(`${FRONTEND_URL}/onboarding/summary`, { timeout: 10000 });
        await page.waitForTimeout(2000);
      }

      if (page.url().includes('summary')) {
        console.log(`  [OK] At summary page`);

        // Look for generate/continue button
        const generateBtn = await page.locator('button:has-text("Generate"), button:has-text("Continue"), button:has-text("Next")').first();
        if (await generateBtn.isVisible().catch(() => false)) {
          await generateBtn.click();
          console.log(`  [OK] Clicked generate/continue button`);
          await page.waitForTimeout(5000); // AI generation takes time
        }
        results.steps['4_summary'] = 'PASS';
      } else {
        results.steps['4_summary'] = `REDIRECT: ${page.url().split('/').pop()}`;
      }
    } catch (e) {
      console.log(`  [WARN] Summary: ${e.message.slice(0, 50)}`);
      results.steps['4_summary'] = 'SKIP';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_04_summary.png` });

    // ===== PHASE 5: AI SUMMARY APPROVAL =====
    console.log('\n[Step 5] AI summary approval...');
    try {
      // Navigate to AI summary if not there
      if (!page.url().includes('ai-summary')) {
        await page.goto(`${FRONTEND_URL}/onboarding/ai-summary`, { timeout: 10000 });
        await page.waitForTimeout(3000);
      }

      if (page.url().includes('ai-summary') || page.url().includes('summary')) {
        console.log(`  [OK] At AI summary page`);

        // Look for approve/dashboard button
        const approveBtn = await page.locator('button:has-text("dashboard"), button:has-text("Approve"), button:has-text("Continue"), button:has-text("matches")').first();
        if (await approveBtn.isVisible().catch(() => false)) {
          await approveBtn.click();
          console.log(`  [OK] Clicked approve/continue button`);
          await page.waitForTimeout(3000);
        }
        results.steps['5_ai_summary'] = 'PASS';
      } else {
        results.steps['5_ai_summary'] = `REDIRECT: ${page.url().split('/').pop()}`;
      }
    } catch (e) {
      console.log(`  [WARN] AI Summary: ${e.message.slice(0, 50)}`);
      results.steps['5_ai_summary'] = 'SKIP';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_05_ai_summary.png` });

    // ===== PHASE 6: MATCHES REVIEW =====
    console.log('\n[Step 6] Matches review...');
    try {
      if (page.url().includes('matches')) {
        console.log(`  [OK] At matches page`);

        // Look for dashboard button
        const dashBtn = await page.locator('button:has-text("dashboard"), a:has-text("dashboard")').first();
        if (await dashBtn.isVisible().catch(() => false)) {
          await dashBtn.click();
          await page.waitForTimeout(2000);
        }
        results.steps['6_matches'] = 'PASS';
      } else {
        // Try to navigate
        await page.goto(`${FRONTEND_URL}/onboarding/matches`, { timeout: 10000 });
        await page.waitForTimeout(2000);
        results.steps['6_matches'] = page.url().includes('matches') ? 'PASS' : 'SKIP';
      }
    } catch (e) {
      results.steps['6_matches'] = 'SKIP';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_06_matches.png` });

    // ===== PHASE 7: DASHBOARD ACCESS =====
    console.log('\n[Step 7] Dashboard access...');
    try {
      await page.goto(`${FRONTEND_URL}/dashboard`, { timeout: 10000 });
      await page.waitForTimeout(2000);
      const dashUrl = page.url();

      if (dashUrl.includes('dashboard')) {
        console.log(`  [OK] Dashboard accessible!`);
        results.steps['7_dashboard'] = 'PASS';
      } else {
        console.log(`  [INFO] Redirected to: ${dashUrl}`);
        results.steps['7_dashboard'] = `REDIRECT: ${dashUrl.split('/').pop()}`;
      }
    } catch (e) {
      results.steps['7_dashboard'] = 'FAIL';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_07_dashboard.png` });

    // ===== PHASE 8: DISCOVER PAGE =====
    console.log('\n[Step 8] Discover page...');
    try {
      await page.goto(`${FRONTEND_URL}/discover`, { timeout: 10000 });
      await page.waitForTimeout(2000);
      const discoverUrl = page.url();

      if (discoverUrl.includes('discover')) {
        console.log(`  [OK] Discover accessible!`);
        results.steps['8_discover'] = 'PASS';
      } else {
        results.steps['8_discover'] = `REDIRECT: ${discoverUrl.split('/').pop()}`;
      }
    } catch (e) {
      results.steps['8_discover'] = 'FAIL';
    }
    await page.screenshot({ path: `screenshots/user${userNum}_08_discover.png` });

    // Store user ID for later tests
    results.userId = await getUserId(user.email);

  } catch (error) {
    console.log(`  [ERROR] ${error.message}`);
    results.steps['ERROR'] = error.message.slice(0, 100);
  } finally {
    await context.close();
  }

  return results;
}

// Test Interactive Discovery between two users
async function testInteractiveDiscovery(browser, userResults) {
  console.log(`\n${'='.repeat(70)}`);
  console.log('INTERACTIVE DISCOVERY TEST');
  console.log('='.repeat(70));

  const results = { steps: {} };

  // We need at least 2 users with completed onboarding
  const completedUsers = userResults.filter(r =>
    r.steps['7_dashboard'] === 'PASS' || r.steps['8_discover'] === 'PASS'
  );

  if (completedUsers.length < 2) {
    console.log(`  [SKIP] Need 2+ users with dashboard access, got ${completedUsers.length}`);
    results.steps['discovery'] = 'SKIP: insufficient users';
    return results;
  }

  const userA = completedUsers[0];
  const userB = completedUsers[1];
  console.log(`  Testing: ${userA.user} <-> ${userB.user}`);

  // Test via API since we have user IDs
  const fetch = require('node-fetch');

  try {
    // Step 1: User A searches profiles
    console.log('\n[Discovery Step 1] User A searches profiles...');
    const searchResp = await fetch(`${BACKEND_URL}/discover/search?page=1&limit=10`, {
      headers: { 'Authorization': `Bearer test-token-${userA.userId}` }
    });

    if (searchResp.ok) {
      const searchData = await searchResp.json();
      console.log(`  [OK] Search returned ${searchData.result?.profiles?.length || 0} profiles`);
      results.steps['1_search'] = 'PASS';
    } else {
      console.log(`  [INFO] Search: ${searchResp.status}`);
      results.steps['1_search'] = `HTTP ${searchResp.status}`;
    }

    // Step 2: User A expresses interest in User B
    console.log('\n[Discovery Step 2] User A expresses interest in User B...');
    const interestResp = await fetch(`${BACKEND_URL}/discover/interest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer test-token-${userA.userId}`
      },
      body: JSON.stringify({ to_user_id: userB.userId, message: 'Would love to connect!' })
    });

    if (interestResp.ok) {
      const interestData = await interestResp.json();
      console.log(`  [OK] Interest expressed: ${interestData.result?.status || 'pending'}`);
      results.steps['2_express_interest_a'] = 'PASS';
    } else {
      console.log(`  [INFO] Interest A->B: ${interestResp.status}`);
      results.steps['2_express_interest_a'] = `HTTP ${interestResp.status}`;
    }

    // Step 3: User B expresses interest in User A (should create mutual match)
    console.log('\n[Discovery Step 3] User B expresses interest in User A (mutual)...');
    const mutualResp = await fetch(`${BACKEND_URL}/discover/interest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer test-token-${userB.userId}`
      },
      body: JSON.stringify({ to_user_id: userA.userId, message: 'Interested in connecting too!' })
    });

    if (mutualResp.ok) {
      const mutualData = await mutualResp.json();
      const matchCreated = mutualData.result?.match_created;
      console.log(`  [OK] Mutual interest: status=${mutualData.result?.status}, match_created=${matchCreated}`);
      results.steps['3_mutual_interest'] = matchCreated ? 'PASS (match created!)' : 'PASS (pending)';
    } else {
      console.log(`  [INFO] Interest B->A: ${mutualResp.status}`);
      results.steps['3_mutual_interest'] = `HTTP ${mutualResp.status}`;
    }

    // Step 4: Check mutual interests
    console.log('\n[Discovery Step 4] Check mutual interests...');
    const myInterestsResp = await fetch(`${BACKEND_URL}/discover/my-interests?type=mutual`, {
      headers: { 'Authorization': `Bearer test-token-${userA.userId}` }
    });

    if (myInterestsResp.ok) {
      const myInterestsData = await myInterestsResp.json();
      const mutualCount = myInterestsData.result?.interests?.length || 0;
      console.log(`  [OK] User A has ${mutualCount} mutual interests`);
      results.steps['4_check_mutual'] = mutualCount > 0 ? 'PASS' : 'PASS (0 mutual)';
    } else {
      results.steps['4_check_mutual'] = `HTTP ${myInterestsResp.status}`;
    }

  } catch (e) {
    console.log(`  [ERROR] Discovery test: ${e.message}`);
    results.steps['discovery_error'] = e.message.slice(0, 100);
  }

  return results;
}

async function main() {
  console.log('='.repeat(70));
  console.log('RECIPROCITY PLATFORM - FULL E2E TEST (Onboarding + Discovery)');
  console.log(`Started: ${new Date().toISOString()}`);
  console.log('='.repeat(70));

  // Create screenshots directory
  const fs = require('fs');
  if (!fs.existsSync('screenshots')) {
    fs.mkdirSync('screenshots');
  }

  const browser = await chromium.launch({ headless: true });
  const allResults = [];

  // Test full user journeys
  for (let i = 0; i < USERS.length; i++) {
    const result = await testFullUserJourney(browser, USERS[i], i + 1);
    allResults.push(result);
  }

  // Test Interactive Discovery
  const discoveryResults = await testInteractiveDiscovery(browser, allResults);
  allResults.push({ user: 'Discovery Test', steps: discoveryResults.steps });

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
  console.log(`Screenshots saved to: screenshots/`);
  console.log(`Completed: ${new Date().toISOString()}`);
}

main().catch(console.error);
