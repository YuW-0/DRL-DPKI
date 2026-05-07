// Connect to the smart contract for reading/writing summary blocks
const { Web3 } = require('web3');

var web3 = new Web3('http://localhost:8545')

// Load NA account private keys from environment variables (comma-separated).
// Example (PowerShell):
//   $env:NA_PRIVATE_KEYS="0x...,0x...,0x...,0x...,0x..."
const privateKeysEnv = process.env.NA_PRIVATE_KEYS || '';
const privateKeys = privateKeysEnv
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

// Add private keys to the web3 wallet
privateKeys.forEach((pk) => {
  web3.eth.accounts.wallet.add(pk);
});

const abi = [
	{
		"inputs": [],
		"stateMutability": "nonpayable",
		"type": "constructor"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": false,
				"internalType": "uint256",
				"name": "totalRequests",
				"type": "uint256"
			}
		],
		"name": "EmergencyRotationActivated",
		"type": "event"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": true,
				"internalType": "address",
				"name": "na",
				"type": "address"
			},
			{
				"indexed": false,
				"internalType": "uint256",
				"name": "totalRequests",
				"type": "uint256"
			}
		],
		"name": "EmergencyRotationRequested",
		"type": "event"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": true,
				"internalType": "address",
				"name": "na",
				"type": "address"
			},
			{
				"indexed": false,
				"internalType": "string",
				"name": "pk",
				"type": "string"
			}
		],
		"name": "NARegistered",
		"type": "event"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": false,
				"internalType": "address payable[5]",
				"name": "newNAs",
				"type": "address[5]"
			},
			{
				"indexed": false,
				"internalType": "string[]",
				"name": "newNAsPK",
				"type": "string[]"
			},
			{
				"indexed": false,
				"internalType": "uint256",
				"name": "newMaxTerm",
				"type": "uint256"
			}
		],
		"name": "NAsUpdated",
		"type": "event"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": true,
				"internalType": "address",
				"name": "na",
				"type": "address"
			},
			{
				"indexed": false,
				"internalType": "string",
				"name": "result",
				"type": "string"
			}
		],
		"name": "TermResultSubmitted",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "NAs",
		"outputs": [
			{
				"internalType": "address payable",
				"name": "",
				"type": "address"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "NAsPK",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "Threshold",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "allNAs",
		"outputs": [
			{
				"internalType": "address payable",
				"name": "",
				"type": "address"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "currentTermCount",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "emergencyRequestCount",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"name": "emergencyRequested",
		"outputs": [
			{
				"internalType": "bool",
				"name": "",
				"type": "bool"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "emergencyRotationActive",
		"outputs": [
			{
				"internalType": "bool",
				"name": "",
				"type": "bool"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "n",
				"type": "uint256"
			}
		],
		"name": "incCurrentTermCount",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "who",
				"type": "address"
			}
		],
		"name": "isCurrentNA",
		"outputs": [
			{
				"internalType": "bool",
				"name": "",
				"type": "bool"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"name": "isRegisteredNA",
		"outputs": [
			{
				"internalType": "bool",
				"name": "",
				"type": "bool"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "maxTerm",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"name": "naNominateCount",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"name": "naPK",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "nominatedCandidates",
		"outputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address payable",
				"name": "naAddr",
				"type": "address"
			},
			{
				"internalType": "string",
				"name": "pk",
				"type": "string"
			}
		],
		"name": "registerNA",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "requestEmergencyRotation",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "result",
				"type": "string"
			}
		],
		"name": "submitTermResult",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"name": "termResult",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"name": "termSubmitted",
		"outputs": [
			{
				"internalType": "bool",
				"name": "",
				"type": "bool"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "termSubmittedCount",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "address payable[5]",
				"name": "newNAs",
				"type": "address[5]"
			},
			{
				"internalType": "string[]",
				"name": "newNAsPK",
				"type": "string[]"
			},
			{
				"internalType": "uint256",
				"name": "newMaxTerm",
				"type": "uint256"
			}
		],
		"name": "updateNAsAndTerm",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	}
]
var contractAddr = '0xDA81FC14B2942C297ca3f287034b28397AAA98D3'
var myContract = new web3.eth.Contract(abi, contractAddr)

module.exports = exdr = {
    web3,
    myContract
}
