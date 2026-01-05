// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;

contract NARotation {
    uint256 public Threshold = 5;
    uint256 public maxTerm = 10;
    uint256 public currentTermCount = 0;
    uint256 public termSubmittedCount = 0;
    address payable[5] public NAs = [
        payable(0xb952c28b5f63Ac983662C4c7EDdbEEA86162d3a8),
        payable(0x50BB5Fbbc02c69B2E133d39fd5F0E6E5Db297A10),
        payable(0x981ba6c681e196B38c65A73573b62Cf43FcCee16),
        payable(0x20339fBE40A1e20b603EEcd57397445a7764ad89),
        payable(0x31602EeBdc7B1489a29e64394C3F24E75B4F3D76)
    ];
    string[] public NAsPK = [
        '0x5a98ee5aab2f41685fdf482ef5bb91dc91d5a2b82b90106c4df4957d333b1f6cefe41a901a0478e8d12dde2ad62f1c941ece8e5b51836f8c700d5e5e6682ab53',
        '0xfe81243a4bebb360878b072589ded6bef9ddcc6b8dfa89598c6e4d7ca353036223393311dc31b5929dfad453152454be085704e9dcf1dc65e049a3df26b2022a',
        '0x0c60e8cfa00bcd46554e2c93c93673aeae3bb107fa4d784cb7b6e7cf790c519f4345dcc0f0a91840856395e3801643129133202037807db5f9bc0454916d4700',
        '0x735fa3ecb356457fd46701814bc17827cc443e076e3328566cc9e88ea9c68136325118f25e8930ae072d3930393d70fd087d2b5d6faf8dff44b778b267c5bfc3',
        '0x31ba30d9158c9641ccdeb24be4daf122fc0c4e81403ce6340d953f140e29f18692c8e12dea0ab28f1e20a492ad59e1e3b0353a67d2e5a94a63e4d84681d4638c'
    ];
    mapping(address => bool) public termSubmitted;
    mapping(address => string) public termResult;
    mapping(address => uint256) public naNominateCount;
    address[] public nominatedCandidates;
    uint256 public emergencyRequestCount = 0;
    bool public emergencyRotationActive = false;
    mapping(address => bool) public emergencyRequested;
    address payable[] public allNAs;
    mapping(address => bool) public isRegisteredNA;
    mapping(address => string) public naPK;

    // Added: low-reputation CAs and malicious CAs
    mapping(address => bool) public lowReputationCAs;
    mapping(address => bool) public maliciousCAs;

    // Added: voting
    struct Vote {
        address[] candidates;
        int256[] qValues;
    }
    mapping(address => Vote) public votes;
    address[] public voters;
    bool public votingPhase = true;
    bool public finalized = false;

    event NAsUpdated(address payable[5] newNAs, string[] newNAsPK, uint256 newMaxTerm);
    event TermResultSubmitted(address indexed na, string result);
    event EmergencyRotationRequested(address indexed na, uint256 totalRequests);
    event EmergencyRotationActivated(uint256 totalRequests);
    event NARegistered(address indexed na, string pk);
    // Added events
    event VoteSubmitted(address indexed voter, address[] candidates, int256[] qValues);
    event VoteViolation(address indexed voter);
    event CAMarkedLowReputation(address ca);
    event CAMarkedMalicious(address ca);

    constructor() {
        for (uint256 i = 0; i < Threshold; i++) {
            address payable naAddr = NAs[i];
            allNAs.push(naAddr);
            isRegisteredNA[naAddr] = true;
            naPK[naAddr] = NAsPK[i];
        }
    }

    function isCurrentNA(address who) public view returns (bool) {
        for (uint256 i = 0; i < Threshold; i++) {
            if (NAs[i] == who) return true;
        }
        return false;
    }

    function submitVote(address[] memory candidates, int256[] memory qValues) public {
        require(isCurrentNA(msg.sender), "only current NA can vote");
        require(!maliciousCAs[msg.sender], "malicious CA cannot vote");
        require(votingPhase, "voting phase closed");
        require(candidates.length == qValues.length, "candidates and qValues length mismatch");
        require(!votes[msg.sender].candidates.length > 0, "already voted"); // Check whether already voted

        // Vote validity constraint: disallow voting for low-reputation CAs
        for (uint256 i = 0; i < candidates.length; i++) {
            require(!lowReputationCAs[candidates[i]], "cannot vote for low reputation CA");
        }

        votes[msg.sender] = Vote(candidates, qValues);
        voters.push(msg.sender);
        emit VoteSubmitted(msg.sender, candidates, qValues);

        // If all NAs have voted, automatically trigger finalizeVoting
        if (voters.length == Threshold) {
            finalizeVoting();
        }
    }

    function updateNAsAndTerm(address payable[5] memory newNAs, string[] memory newNAsPK, uint256 newMaxTerm) public {
        require(isCurrentNA(msg.sender), "only current NA can trigger update");
        require(currentTermCount >= (maxTerm - 1) * Threshold, "term not ending yet");
        require(termSubmittedCount >= Threshold, "not all NAs submitted term results");
        require(newNAsPK.length == Threshold, "NAsPK length must match Threshold");
        require(newMaxTerm > 0, "maxTerm must be positive");
        address payable[5] memory oldNAs = NAs;
        for (uint256 i = 0; i < Threshold; i++) {
            termSubmitted[oldNAs[i]] = false;
            termResult[oldNAs[i]] = "";
            emergencyRequested[oldNAs[i]] = false;
        }
        NAs = newNAs;
        NAsPK = newNAsPK;
        currentTermCount = 0;
        maxTerm = newMaxTerm;
        termSubmittedCount = 0;
        emergencyRequestCount = 0;
        emergencyRotationActive = false;
        emit NAsUpdated(newNAs, newNAsPK, newMaxTerm);
    }

    function requestEmergencyRotation() public {
        require(!emergencyRotationActive, "emergency rotation already active");
        require(isCurrentNA(msg.sender), "only current NA can request emergency rotation");
        require(!maliciousCAs[msg.sender], "malicious CA cannot request");
        require(!emergencyRequested[msg.sender], "already requested");
        emergencyRequested[msg.sender] = true;
        emergencyRequestCount += 1;
        emit EmergencyRotationRequested(msg.sender, emergencyRequestCount);

        uint256 required = (Threshold * 2) / 3 + 1;
        if (emergencyRequestCount >= required) {
            _activateEmergencyRotation();
        }
    }

    function _activateEmergencyRotation() internal {
        if (emergencyRotationActive) {
            return;
        }
        emergencyRotationActive = true;
        for (uint256 i = 0; i < Threshold; i++) {
            termSubmitted[NAs[i]] = false;
            termResult[NAs[i]] = "";
        }
        for (uint256 j = 0; j < nominatedCandidates.length; j++) {
            naNominateCount[nominatedCandidates[j]] = 0;
        }
        delete nominatedCandidates;
        termSubmittedCount = 0;
        currentTermCount = (maxTerm - 1) * Threshold;
        emit EmergencyRotationActivated(emergencyRequestCount);
    }

    function registerNA(address payable naAddr, string memory pk) public {
        require(isCurrentNA(msg.sender), "only current NA can register");
        require(bytes(pk).length != 0, "pk required");
        if (!isRegisteredNA[naAddr]) {
            allNAs.push(naAddr);
            isRegisteredNA[naAddr] = true;
        }
        naPK[naAddr] = pk;
        emit NARegistered(naAddr, pk);
    }

    function parseAddressList(string memory s) internal pure returns (address[] memory) {
        address[] memory addrs = new address[](5);
        bytes memory strBytes = bytes(s);
        uint256 last = 0;
        uint256 count = 0;
        for (uint256 i = 0; i < strBytes.length; i++) {
            if (strBytes[i] == ',') {
                addrs[count] = parseAddress(slice(s, last, i));
                count++;
                last = i + 1;
            }
        }
        addrs[count] = parseAddress(slice(s, last, strBytes.length));
        return addrs;
    }

    function slice(string memory s, uint256 start, uint256 end) internal pure returns (string memory) {
        bytes memory strBytes = bytes(s);
        require(end <= strBytes.length, "slice: end out of bounds");
        require(start < end, "slice: invalid range");
        bytes memory result = new bytes(end - start);
        for (uint256 i = start; i < end; i++) {
            result[i - start] = strBytes[i];
        }
        return string(result);
    }

    function parseAddress(string memory s) internal pure returns (address) {
        bytes memory tmp = bytes(s);
        // Length check: must be 42 characters (0x + 40 hex)
        require(tmp.length == 42, "Invalid address format: must be 42 chars (0x + 40 hex)");
        // Verify prefix is "0x"
        require(tmp[0] == '0' && (tmp[1] == 'x' || tmp[1] == 'X'), "Address must start with 0x");
        
        uint160 iaddr = 0;
        for (uint256 i = 2; i < 42; i++) {
            uint8 b = uint8(tmp[i]);
            uint8 v;
            if (b >= 97 && b <= 102) v = b - 87;      // a-f
            else if (b >= 65 && b <= 70) v = b - 55;  // A-F
            else if (b >= 48 && b <= 57) v = b - 48;  // 0-9
            else revert("Invalid hex character");
            iaddr = (iaddr << 4) | v;
        }
        return address(iaddr);
    }

    // ====== Test helper: manually increment currentTermCount ======
    function incCurrentTermCount(uint256 n) public {
        currentTermCount += n;
    }

    // Added: consistency check and finalizeVoting
    function finalizeVoting() internal {
        require(votingPhase, "voting not active");
        require(voters.length == Threshold, "not all NAs voted");
        votingPhase = false;
        finalized = true;

        // Collect all unique candidate CAs
        address[] memory allCandidates = collectUniqueCandidates();

        // Perform consistency checks for each candidate
        for (uint256 i = 0; i < allCandidates.length; i++) {
            address candidate = allCandidates[i];
            int256[] memory qValuesForCandidate = getAllQValuesForCandidate(candidate);
            if (qValuesForCandidate.length == 0) continue;

            int256 median = calculateMedian(qValuesForCandidate);

            // Check whether each voter's Q-value is within ±10% of the median
            for (uint256 j = 0; j < voters.length; j++) {
                address voter = voters[j];
                int256 voterQ = getVoterQForCandidate(voter, candidate);
                if (voterQ != 0) {  // Only check when the voter actually voted for the candidate
                    int256 lowerBound = median * 9 / 10;
                    int256 upperBound = median * 11 / 10;
                    if (voterQ < lowerBound || voterQ > upperBound) {
                        // Violation: emit event
                        emit VoteViolation(voter);
                        // Mark as low-reputation; if already low-reputation, mark as malicious
                        if (!lowReputationCAs[voter]) {
                            lowReputationCAs[voter] = true;
                            emit CAMarkedLowReputation(voter);
                        } else if (!maliciousCAs[voter]) {
                            maliciousCAs[voter] = true;
                            emit CAMarkedMalicious(voter);
                        }
                    }
                }
            }
        }

        // Continue with the original flow: select new NAs
        selectNewNAs();
    }

    function collectUniqueCandidates() internal view returns (address[] memory) {
        address[] memory temp = new address[](voters.length * 5); // Assume up to 5 candidates per vote
        uint256 count = 0;
        for (uint256 i = 0; i < voters.length; i++) {
            Vote memory vote = votes[voters[i]];
            for (uint256 j = 0; j < vote.candidates.length; j++) {
                address cand = vote.candidates[j];
                bool exists = false;
                for (uint256 k = 0; k < count; k++) {
                    if (temp[k] == cand) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    temp[count] = cand;
                    count++;
                }
            }
        }
        address[] memory result = new address[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = temp[i];
        }
        return result;
    }

    function getVoterQForCandidate(address voter, address candidate) internal view returns (int256) {
        Vote memory vote = votes[voter];
        for (uint256 i = 0; i < vote.candidates.length; i++) {
            if (vote.candidates[i] == candidate) {
                return vote.qValues[i];
            }
        }
        return 0; // Voter did not vote for this candidate
    }

    function getAllQValuesForCandidate(address candidate) internal view returns (int256[] memory) {
        int256[] memory qValues = new int256[](voters.length);
        uint256 count = 0;
        for (uint256 i = 0; i < voters.length; i++) {
            int256 q = getVoterQForCandidate(voters[i], candidate);
            if (q != 0) {
                qValues[count] = q;
                count++;
            }
        }
        // Truncate to valid portion
        int256[] memory result = new int256[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = qValues[i];
        }
        return result;
    }

    function calculateMedian(int256[] memory arr) internal pure returns (int256) {
        if (arr.length == 0) return 0;
        int256[] memory sortedArr = sort(arr);
        uint256 mid = sortedArr.length / 2;
        if (sortedArr.length % 2 == 0) {
            return (sortedArr[mid - 1] + sortedArr[mid]) / 2;
        } else {
            return sortedArr[mid];
        }
    }

    function sort(int256[] memory arr) internal pure returns (int256[] memory) {
        int256[] memory result = new int256[](arr.length);
        for (uint256 i = 0; i < arr.length; i++) {
            result[i] = arr[i];
        }
        for (uint256 i = 0; i < result.length; i++) {
            for (uint256 j = i + 1; j < result.length; j++) {
                if (result[i] > result[j]) {
                    int256 temp = result[i];
                    result[i] = result[j];
                    result[j] = temp;
                }
            }
        }
        return result;
    }

    function selectNewNAs() internal {
        // Simplified: select candidates with the most votes
        address[] memory candidates = collectUniqueCandidates();
        uint256[] memory voteCounts = new uint256[](candidates.length);
        for (uint256 i = 0; i < candidates.length; i++) {
            for (uint256 j = 0; j < voters.length; j++) {
                if (getVoterQForCandidate(voters[j], candidates[i]) != 0) {
                    voteCounts[i]++;
                }
            }
        }

        address payable[5] memory newNAs;
        string[] memory newNAsPK = new string[](Threshold);
        for (uint256 t = 0; t < Threshold; t++) {
            uint256 maxVotes = 0;
            address maxAddr = address(0);
            for (uint256 j = 0; j < candidates.length; j++) {
                if (voteCounts[j] > maxVotes) {
                    maxVotes = voteCounts[j];
                    maxAddr = candidates[j];
                }
            }
            if (maxAddr == address(0)) break;
            newNAs[t] = payable(maxAddr);
            require(isRegisteredNA[maxAddr], "candidate not registered");
            string memory pk = naPK[maxAddr];
            require(bytes(pk).length != 0, "candidate PK missing");
            newNAsPK[t] = pk;
            // Remove selected candidate
            for (uint256 k = 0; k < candidates.length; k++) {
                if (candidates[k] == maxAddr) {
                    voteCounts[k] = 0;
                    break;
                }
            }
        }
        updateNAsAndTerm(newNAs, newNAsPK, maxTerm);
    }
}
