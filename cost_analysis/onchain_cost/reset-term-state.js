const { web3, myContract } = require('./NARotationContract');

const NA_ACCOUNTS = [
  '0xb952c28b5f63Ac983662C4c7EDdbEEA86162d3a8',
  '0x50BB5Fbbc02c69B2E133d39fd5F0E6E5Db297A10',
  '0x981ba6c681e196B38c65A73573b62Cf43FcCee16',
  '0x20339fBE40A1e20b603EEcd57397445a7764ad89',
  '0x31602EeBdc7B1489a29e64394C3F24E75B4F3D76'
];

async function ensureTermCounter() {
  const maxTerm = Number(await myContract.methods.maxTerm().call());
  const threshold = NA_ACCOUNTS.length;
  const target = (maxTerm - 1) * threshold;
  const current = Number(await myContract.methods.currentTermCount().call());

  if (current < target) {
    console.log(`Incrementing currentTermCount from ${current} to ${target}`);
    await myContract.methods
      .incCurrentTermCount(target - current)
      .send({ from: NA_ACCOUNTS[0], gas: 1_000_000 });
  } else {
    console.log(`currentTermCount already >= target (${current} >= ${target})`);
  }
}

async function submitMissingResults() {
  const resultStr = NA_ACCOUNTS.join(',');
  for (const na of NA_ACCOUNTS) {
    const submitted = await myContract.methods.termSubmitted(na).call();
    console.log(`NA ${na} submitted: ${submitted}`);
    if (submitted) {
      continue;
    }
    console.log(`Submitting term result for ${na}`);
    const gasEstimate = await myContract.methods
      .submitTermResult(resultStr)
      .estimateGas({ from: na });
    const safeGas = Math.floor(Number(gasEstimate) * 3);
    const receipt = await myContract.methods
      .submitTermResult(resultStr)
      .send({ from: na, gas: safeGas });
    console.log(
      `  OK tx ${receipt.transactionHash} gasUsed ${receipt.gasUsed}`
    );
  }
}

async function showFinalState() {
  const termCount = Number(await myContract.methods.termSubmittedCount().call());
  const currentTerm = Number(await myContract.methods.currentTermCount().call());
  const emergency = await myContract.methods.emergencyRotationActive().call();
  console.log('--- Final State ---');
  console.log('termSubmittedCount:', termCount);
  console.log('currentTermCount:', currentTerm);
  console.log('emergencyRotationActive:', emergency);
}

async function main() {
  try {
    await ensureTermCounter();
    await submitMissingResults();
    await showFinalState();
  } catch (err) {
    console.error('Reset failed:', err.message || err);
  }
  process.exit(0);
}

main();
