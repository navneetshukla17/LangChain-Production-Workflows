from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load gemini api key
load_dotenv()

# Model configuration
embedding_model = GoogleGenerativeAIEmbeddings(
    model='models/text-embedding-004',
    dimensions=300
)

# Sample document - for semantic search
document = """

# Cricketers and Their Specialities (List Format)

1. **Virat Kohli**

   * Master of run chases
   * Exceptional consistency across formats
   * Elite fitness and fielding

2. **MS Dhoni**

   * Best finisher in limited-overs cricket
   * Legendary captaincy and calmness
   * Lightning-fast wicketkeeping

3. **Rohit Sharma**

   * Known for multiple ODI double centuries
   * Elegant timing and effortless power
   * Big-match player

4. **Jasprit Bumrah**

   * Yorker specialist
   * Unorthodox action with high accuracy
   * Match-winner across formats

5. **Ben Stokes**

   * Clutch performer in crucial matches
   * Dominant all-rounder
   * Fearless mentality

6. **AB de Villiers**

   * Mr. 360° shot-making
   * Creative and destructive batting style
   * Exceptional finisher

7. **Rashid Khan**

   * World-class leg spinner
   * Deadly googly and quick variations
   * Useful lower-order hitter

8. **Kane Williamson**

   * Technically strong batsman
   * Calm and composed leader
   * Excellent performer in tough conditions

9. **Shaheen Shah Afridi**

   * Lethal inswinging yorkers
   * Aggressive new-ball wicket-taker
   * High pace with steep bounce

10. **Steve Smith**

* Unorthodox but highly effective technique
* Remarkably consistent in Tests
* Exceptional concentration

"""

# start from here


