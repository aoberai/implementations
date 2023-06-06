'''
I know very little about how blockchains work as of mid 2023 so here are some notes as I learn shit. 

Sk = private key = secret key
Pk = public key

Every transaction on a ledger needs a signature -- this signature is a combination of the message, the index, and the private key -- unforgeable because only you have the private key.
Message includes index
Sign(Message, sk) = Signature --> this is a one-way function -- aka digital signature
Verify(Message, Signature, pk) = T/F -- is signature associated with public key
Message is what the desired transaction is encoded in some format
You can't guess a signature without secret key

Verification of a message from a public key with a given signature, you can be confident the only way that signature could be created is if they knew the secret key for the public key. This makes things complete with the sign function because the signature, just a set of binary numbers, is not validity for the transaction. Need to learn how the verify function works
If I want to add a line to this shared ledger, I can simply write a message detailing a transaction, provide my secret key to create a signature with the message. Anyone else can see that I have approved the transaction with the Verify function using my public key.

Transactions only approved if money available. You don't revert back to cash.

Bitcoin is actually the ledger -- the history of transactions.

Instead of a centralized place the ledger is kept, everyone has their own. When a new transaction is made, its broadcasted and everyone adds it to the chain. Question: How is it broadcasted? You want all these different ledgers to look the same

There is a chance someone else doesn't record the correct transaction

Proof of Work:

Computational work

Some number + the ledger, when hashed, is hoping for x number of 0's in the beginning of the hash. You can increase the number of 0's to increase the amount of time it takes to search for that random number. 

You need to be able to agree on the right ledger. Organize ledgers into a given block. Each block has a list of transactions plus a proof of work such that the hash starts with x number of 0's. 

A transaction is only considered valid with a signature from the sender, block only valid if it has proof of work. Order of blocks such that the block has the previous hash of the previous block. You would have to redo all the work if you change the hashes around. This composes the block chain. 

Listen to transactions, collect them into a block, know previous block hash to include, do some work to do proof of work. Broadcast the block when you succcesfully do the work. If you do the work properly, you get block reward. Doing proof of work is totally luck based

Two conflicting block chains with differing transactions, take the longest chain (the one which has the most work put into it)

If you add a incorrect block to one chain (she finds proof of work before everyone else), it still wont necessarily be trusted since there can be a branching factor where one block has two children and those children grow and the real chain, supported by a majority of the compute, grows longer because it has more compute available to grow the chain and find proof of work many times faster. The longest chain is the trusted one. 

Different block time for a group of miners to find hash. Bitcoin aims for 1 every 10 minutes

Reward per block decreases geometrically -- wont be more than 21 million in existance

Miners will still continue because they receive transaction fees that goes straight to the miner.

Incentivizes you to include a transaction into a block. This is like any other transaction requiring a signature from the sender to make a transaction. This is also referred to as the gas fees (which is really high for ethereum).

Each block limited to ~2400 transactions

The argument of number of transactions per block is the block size debate?

The reason proof of work ensures validity is because it gives order to blocks. Fake blocks can't be added anyways because you have to regenerate a whole new chain which means new hashes and your whole things diverges.

Okay this makes sense cool algorithm

Other big questions:

Where do smart contracts run and how are they guaranteed?
How do you peg to the dollar like in USDC or peg to prices of gold?
Distributed compute for other applications?


'''



