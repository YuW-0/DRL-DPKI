// NA/na-batch-rotation.js
// Simulate NA batch submission of submitTermResult (term rotation).
// Runs multiple rounds and reports average execution time and gas usage.
// Uses web3.js. Deploy the contract first and configure ABI and contract address.

const exrr = require('./NARotationContract');
const contract = exrr.myContract;
const web3 = exrr.web3;

// Extend web3 with debug_traceCall support to measure pure contract execution time.
if (typeof web3.eth.traceCall !== 'function') {
  web3.eth.extend({
    methods: [
      {
        name: 'traceCall',
        call: 'debug_traceCall',
        params: 3
      }
    ]
  });
}
// NA accounts (recommended: load their private keys into web3.eth.accounts.wallet)
const NA_accounts = [
  '0xb952c28b5f63Ac983662C4c7EDdbEEA86162d3a8',
  '0x50BB5Fbbc02c69B2E133d39fd5F0E6E5Db297A10',
  '0x981ba6c681e196B38c65A73573b62Cf43FcCee16',
  '0x20339fBE40A1e20b603EEcd57397445a7764ad89',
  '0x31602EeBdc7B1489a29e64394C3F24E75B4F3D76'
];

// ====== Main flow ======
async function main() {

  let totalGas = 0;
  const rounds = 30;
  const Threshold = NA_accounts.length;
  
  // Arrays for statistics
  const perRoundTimes = [];  // Per-round NA execution times

  // Fetch maxTerm
  const maxTerm = parseInt(await contract.methods.maxTerm().call());
  
  for (let round = 0; round < rounds; round++) { // Run multiple rounds
    console.log(`\n=== Rotation Round ${round + 1} ===`);
    
    // Check current rotation state
    let termSubmittedCount = parseInt(await contract.methods.termSubmittedCount().call());
    console.log(`Submitted NA count for current rotation: ${termSubmittedCount}`);
    
    // If some NAs already submitted, finalize/reset the current rotation first.
    if (termSubmittedCount > 0) {
      console.log('Detected partial submissions. Forcing state reset...');
      
      // Fetch contract parameters
      const currentTerm = await contract.methods.currentTermCount().call();
      const maxTerm = await contract.methods.maxTerm().call();
      const threshold = await contract.methods.Threshold().call();
      
      // Adjust term counter to satisfy rotation condition
      const targetTerm = (parseInt(maxTerm) - 1) * parseInt(threshold);
      if (parseInt(currentTerm) < targetTerm) {
        console.log(`Adjusting term counter from ${currentTerm} to ${targetTerm}`);
        await contract.methods.incCurrentTermCount(targetTerm - parseInt(currentTerm)).send({
          from: NA_accounts[0], 
          gas: 1000000
        });
      }
      
      // Force rotation update to reset all NA state
      console.log('Triggering rotation update to reset all NA state...');
      try {
        await contract.methods.updateNAsAndTerm(
          NA_accounts,
          ['pk1', 'pk2', 'pk3', 'pk4', 'pk5'],
          parseInt(maxTerm)
        ).send({from: NA_accounts[0], gas: 5000000});
        console.log('Rotation update completed. All NA state has been reset.');
      } catch (error) {
        console.log('Rotation update failed:', error.message);
        console.log('Skipping this round.');
        perRoundTimes.push([]);
        continue;
      }
      
      // Wait for state to stabilize
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    // Before each round, manually increment currentTermCount to satisfy rotation condition.
    await contract.methods.incCurrentTermCount((maxTerm - 1) * Threshold).send({from: NA_accounts[0]});
    
    // Simulate each NA submitting submitTermResult using the current NA list.
    const resultStr = NA_accounts.join(',');
    
    // Ensure all NAs are in "not submitted" state
    const allUnsubmitted = await Promise.all(
      NA_accounts.map(na => contract.methods.termSubmitted(na).call())
    );
    
    if (allUnsubmitted.some(submitted => submitted)) {
      console.log('Warning: some NAs are still marked as submitted. Skipping this round.');
      perRoundTimes.push([]);
      continue;
    }
    // Submit transactions serially, one NA at a time
    const receipts = [];
    let gasSum = 0;
    const roundNATimes = []; // Per-NA execution time for this round
    
    for (let i = 0; i < NA_accounts.length; i++) {
      const na = NA_accounts[i];
      console.log(`  Submitting NA ${i + 1}/${NA_accounts.length}: ${na}`);

      const method = contract.methods.submitTermResult(resultStr);

      let gasEstimate;
      try {
        gasEstimate = await method.estimateGas({from: na});
      } catch (error) {
        console.log(`    ERROR: estimateGas failed: ${error.message}`);
        continue;
      }

      let naTimeMs = Number.NaN;
      const traceGas = Math.floor(Number(gasEstimate) * 2);
      const traceConfig = { tracer: 'callTracer', timeout: '30s' };
      const traceCallPayload = {
        from: na,
        to: contract.options.address,
        data: method.encodeABI(),
        gas: web3.utils.toHex(traceGas)
      };

      try {
        const naStartTime = process.hrtime.bigint();
        await web3.eth.traceCall(traceCallPayload, 'latest', traceConfig);
        const naEndTime = process.hrtime.bigint();
        naTimeMs = Number(naEndTime - naStartTime) / 1000000;
      } catch (traceError) {
        console.log(`    WARN: traceCall failed; cannot measure pure execution time: ${traceError.message}`);
      }

      try {
        const safeGas = Math.floor(Number(gasEstimate) * 3);
        const receipt = await method.send({from: na, gas: safeGas});
        roundNATimes.push(naTimeMs);
        receipts.push(receipt);
        gasSum += Number(receipt.gasUsed);

        if (Number.isFinite(naTimeMs)) {
          console.log(`    OK: gas=${receipt.gasUsed}, execTime=${naTimeMs.toFixed(2)}ms`);
        } else {
          console.log(`    OK: gas=${receipt.gasUsed}`);
        }
      } catch (sendError) {
        console.log(`    ERROR: transaction send failed: ${sendError.message}`);
      }
    }
    
    totalGas += gasSum;
    
    // Show per-round breakdown
    perRoundTimes.push(roundNATimes);

    const displayTimes = roundNATimes.map(t => Number.isFinite(t) ? `${t.toFixed(2)}ms` : 'N/A');
    console.log(`Round breakdown:`);
    console.log(`  Per-NA exec times: ${displayTimes.join(', ')}`);
    console.log(`  Total gas this round: ${gasSum}`);
  }
  
  // Calculate per-round average NA execution time
  const roundAvgNATimes = perRoundTimes.map(times => {
    const validTimes = times.filter(Number.isFinite);
    if (validTimes.length === 0) {
      return 0;
    }
    return validTimes.reduce((sum, time) => sum + time, 0) / validTimes.length;
  });

  const allNATimes = perRoundTimes.flat().filter(Number.isFinite);
  const avgNATime = allNATimes.length
    ? allNATimes.reduce((sum, time) => sum + time, 0) / allNATimes.length
    : 0;
  
  console.log(`\n=== Final Summary ===`);
  console.log(`Average gas per round (${rounds} rounds): ${Math.round(totalGas/rounds)}`);
  console.log(`Average per-NA execution time: ${avgNATime.toFixed(2)}ms`);
  console.log(`Average per-NA execution time: ${avgNATime.toFixed(2)}ms (on-chain execution only)`);
  console.log(`NA count: ${NA_accounts.length}`);
  console.log(`Total rounds: ${rounds}`);
  console.log(`Total NA executions: ${allNATimes.length}`);
  
  // Export data for chart generation
  const chartData = {
    rounds: rounds,
    naCount: NA_accounts.length,
    roundNumbers: Array.from({length: rounds}, (_, i) => i + 1),
    avgNATimes: roundAvgNATimes,
    averages: {
      naTime: avgNATime
    }
  };
  
  // Write JSON file
  const fs = require('fs');
  const path = require('path');
  const outputPath = path.join(__dirname, 'performance-data.json');
  fs.writeFileSync(outputPath, JSON.stringify(chartData, null, 2));
  console.log(`\nData exported to: ${outputPath}`);
}

main().catch(console.error);
