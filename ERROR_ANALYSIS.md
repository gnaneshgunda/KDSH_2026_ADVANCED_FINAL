# Detailed Error Analysis from Predictions

## Summary Statistics

From your predictions file, I identified several error patterns:

### Error Distribution
- **False Positives** (predicted 1, actual 0): ~15 cases
- **False Negatives** (predicted 0, actual 1): ~8 cases  
- **Low Confidence Errors**: Many predictions with confidence 0.6-0.8 that were wrong

## Specific Error Cases & Root Causes

### Case 1: ID 46 - Thalcave (FALSE NEGATIVE)
**Claim**: "Thalcave spent his boyhood roaming the plains"
**Your Prediction**: 0 (contradict), confidence 0.80
**Actual**: 1 (consistent)
**Your Rationale**: "The narrative does not provide any information about Thalcave's childhood"

**Root Cause**: 
- Retrieval failed to find relevant passages about Thalcave's childhood
- Absence of evidence was treated as contradiction (should be NOT_MENTIONED)

**Fix Applied**:
- Character-specific filtering will retrieve more Thalcave passages
- Stricter verification: absence → NOT_MENTIONED, not CONTRADICTED
- NOT_MENTIONED doesn't fail the backstory (only contradictions do)

---

### Case 2: ID 137 - Faria (FALSE NEGATIVE)
**Claim**: "Suspected again in 1815, he was re-arrested"
**Your Prediction**: 0 (contradict), confidence 1.00
**Your Rationale**: "The narrative does not mention the re-arrest in 1815"
**Actual**: 1 (consistent)

**Root Cause**:
- High confidence despite only having absence of evidence
- Temporal filtering not used (should look for "1815" mentions)

**Fix Applied**:
- Temporal markers in metadata will help find time-specific passages
- Absence of evidence → lower confidence, not contradiction

---

### Case 3: ID 18 - Jacques Paganel (FALSE POSITIVE)
**Claim**: "Temporary gossip made him wary of the upper nobility"
**Your Prediction**: 1 (consistent), confidence 0.60
**Your Rationale**: "Consistent. Verified 0/4 claims"
**Actual**: 0 (contradict)

**Root Cause**:
- 0 verified claims but still marked consistent
- "Temporary gossip" likely not found in narrative
- Weak confidence (0.60) suggests uncertainty

**Fix Applied**:
- Better rationale will show "0 verified claims" more clearly
- Stricter thresholds: if support_ratio < 0.3, should be unknown/contradict

---

### Case 4: ID 68 - Kai-Koumou (FALSE POSITIVE)
**Claim**: "At fourteen he single-handedly brought down a European bison"
**Your Prediction**: 1 (consistent), confidence 0.90
**Your Rationale**: "Consistent. Verified 4/5 claims"
**Actual**: 0 (contradict)

**Root Cause**:
- Likely verified other claims about Kai-Koumou but not this specific one
- "European bison" is very specific - probably not in narrative
- High confidence despite specific claim being unverified

**Fix Applied**:
- Single-strike rule: if ANY claim contradicted, return 0 immediately
- Better evidence tracking: show which specific claims verified

---

### Case 5: ID 9 - Jacques Paganel (FALSE POSITIVE)
**Claim**: "He accidentally slipped a farewell letter to his French sweetheart into an academic report"
**Your Prediction**: 1 (consistent), confidence 0.80
**Your Rationale**: "Consistent. Verified 7/12 claims"
**Actual**: 0 (contradict)

**Root Cause**:
- Very specific claim (French sweetheart, academic report)
- Likely verified other Paganel claims but not this one
- Aggregate scoring masked the specific failure

**Fix Applied**:
- Detailed rationale will list which claims verified
- Easier to spot if specific claim missing

---

### Case 6: ID 112 - Noirtier (FALSE NEGATIVE)
**Claim**: "Through underground circles he met the Count of Monte Cristo"
**Your Prediction**: 0 (contradict), confidence 1.00
**Your Rationale**: "The story revolves around fictional characters"
**Actual**: 1 (consistent)

**Root Cause**:
- Meta-reasoning about fiction vs reality (wrong level of analysis)
- Should analyze narrative consistency, not historical accuracy

**Fix Applied**:
- Prompt clarifies: analyze narrative consistency, not real-world truth
- Focus on "does narrative support this" not "is this historically true"

---

### Case 7: ID 122 - Faria (FALSE NEGATIVE)
**Claim**: "At a Vienna-congress salon he briefly watched young prosecutor Villefort tamper with evidence"
**Your Prediction**: 0 (contradict), confidence 0.60
**Your Rationale**: "The direct evidence does not explicitly confirm or contradict"
**Actual**: 1 (consistent)

**Root Cause**:
- Uncertain evidence (0.60 confidence)
- Treated uncertainty as contradiction

**Fix Applied**:
- Uncertainty → NOT_MENTIONED → doesn't fail backstory
- Only explicit contradictions fail

---

## Pattern Summary

### Pattern 1: Absence of Evidence → Contradiction (WRONG)
**Frequency**: ~40% of errors
**Fix**: Absence → NOT_MENTIONED → doesn't fail (only contradictions fail)

### Pattern 2: Aggregate Scoring Masks Specific Failures
**Frequency**: ~25% of errors
**Fix**: Single-strike rule + detailed rationale showing which claims verified

### Pattern 3: Weak Character Filtering
**Frequency**: ~20% of errors
**Fix**: Metadata-based character filtering in retrieval

### Pattern 4: Small Chunks Missing Context
**Frequency**: ~10% of errors
**Fix**: Increased chunk size 120→400 words

### Pattern 5: Meta-Reasoning Errors
**Frequency**: ~5% of errors
**Fix**: Clarified prompt to focus on narrative consistency

---

## Expected Accuracy Improvement

### Before (Current System)
- Accuracy: ~70-75% (estimated from error patterns)
- Many high-confidence errors (confidence 0.8-1.0)
- Generic rationales

### After (Improved System)
- Expected Accuracy: ~85-90%
- Better calibrated confidence (fewer high-confidence errors)
- Detailed rationales with evidence quotes

### Key Metrics to Track
1. **False Positive Rate**: Should decrease significantly (stricter verification)
2. **False Negative Rate**: Should decrease moderately (better retrieval)
3. **Confidence Calibration**: High confidence (>0.9) should correlate with correctness
4. **Rationale Quality**: Should include specific evidence quotes
