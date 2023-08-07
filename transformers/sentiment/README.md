this is working pretty well, as of Aug 7 2023. does accurately with sentences like "I don't think it was that good" (as negative sentiment) but still often struggles with the opposite of "i don't think it was that bad" -- just because there are so many "negative" words like "don't" and "bad". Very accurate at simple examples that contain positive or negative adjectives or adverbs and it does seem to convey uncertainty about neutral words (ex: "mid") or paired opposite words (ex: "good but bad"). This might be better by the time you are reading this because I forgot to update the readme

I'm sure hyperparameter changes and better regularization would significantly improve validation performance, I didn't really tinker much since I don't have the hardware to rapidly experiment. lots of obvious fixes that can be made. Did this to get a thorough understanding of how attention and transformers enable llms to be good, which was achieved. 


