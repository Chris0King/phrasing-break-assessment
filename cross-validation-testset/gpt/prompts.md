# Promts
## zero-shot prompt
```
Please assess the phrase break in a speaker's speech from 1-3. 
Our rubric is: Score 1 indicates that many phrase breaks in the sentence are inappropriate and detract from the listener’s understanding. Score 2 indicates that the sentence is fluent, although there are some inappropriate breaks. Score 3 indicates that the sentence is fluent and all phrase breaks are appropriate. 
Now, a speech clip is processed into token sequence omitting punctuation. The token sequence consists of words and break tokens, represented by <0>, <1>, <2>, and <3>. The symbol <0> indicates a duration of less than 10ms between adjacent words. The symbol <1> indicates a duration of 10ms to 50ms between adjacent words. The symbol <2> indicates a duration of 50ms to 200ms between adjacent words. The symbol <3> indicates a duration of over 200ms between adjacent words.
For example, "For example, the token sequence "Good <0> morning <1> Mr. <2> Blue" corresponds to the sentence "Good morning, Mr. Blue". This indicates that the duration between "good" and "morning" is less than 10ms, and the duration between "morning" and "Mr." is between 10ms and 50ms. 
Additionally, your response should only contain a score and a list of inappropriate phrase breaks if there are. An output may look like: Score: 1/2/3, Inappropriate Breaks: ["Mr. <2> Blue"]. 
Please assess the following input: 
```

## few-shot prompts
### few-shot for testset_0
```
...
Here are more examples that you can refer to:
example1:
input: but <2> the <0> mouse <1> finally <1> got <1> out <0> of <0> her <1> mouth <1> and <0> ran <1> away <2> since <0> she <0> could <0> not <0> bite <0> it
output: Score:2, Inappropriate Breaks: ["mouse <1> finally", "finally <1> got", "her <1> mouth", “away <2> since”].
example2:
input: one <0> day <1> a <1> crow <0> stood <1> on <1> a <3> branch <0> near <0> his <0> nest <1> and <3> felt <0> very <0> happy <1> with <0> the <3> meat <2> in <1> his <0> mouth
output: Score:1, Inappropriate Breaks: [“a <1> crow”, “a <3> branch”, “branch <0> near”, “the <3> meat”, “meat <2> in”].
example3:
input: dad <1> he <0> puffed <3> is <0> it <1> true <1> that <1> an <1> apple <1> a <0> day <1> keeps <0> the <0> doctor <0> away
output: Score:3, Inappropriate Breaks: [].
Example4:
input: his <0> mother <0> noticed <1> his <0> left <0> shoe <1> was <0> on <0> his <0> right <0> foot
output: Score:3, Inappropriate Breaks: [].
...
```

### few-shot for testset_1
```
Here are more examples that you can refer to:
example1:
input: the <0> man <0> replied <2> i <0> did <1> we <0> had <0> such <0> a <0> good <0> time <1> we <0> are <0> going <0> to <0> the <0> beach <0> this <0> weekend
output: Score: 3, Inappropriate Breaks: [].
example2:
input: one <0> day <1> a <0> crow <0> stood <1> on <0> a <0> branch <1> near <0> his <0> nest <3> and <0> felt <1> very <0> happy <1> with <0> the <0> meat <1> in <0> his <0> mouth
output: Score:3, Inappropriate Breaks: [].
example3:
input: one <2> afternoon <1> he <0> thought <1> up <1> a <0> good <0> plan <0> to <0> have <0> fun
output: Score:3, Inappropriate Breaks: [“afternoon <1> he”].
Example4:
input: until <1> the <0> fox <3> thought <0> highly <3> of <3> the <0> crow's <0> beautiful <0> voice <1> the <0> crow <0> felt <1> quite <3> proud <1> and <0> finally <0> opened <3> his <0> mouth <0> to <3> sing
output: Score:2, Inappropriate Breaks: [“fox <3> thought”, “highly <3> of”, “quite <3> proud”, opened <3> his, “to <3> sing”].
```

### few-shot for testset_2
```
Here are more examples that you can refer to:
example1:
input: the <0> frogs <0> in <0> the <0> pond <0> were <0> very <0> afraid <0> of <0> the <0> children <0> because <0> the <0> stones <0> killed <0> some <1> of <0> them
output: Score: 3, Inappropriate Breaks: [].
example2:
input: the <0> frogs <0> in <0> the <0> pond <0> were <0> very <0> afraid <0> of <0> the <0> children <1> because <0> the <0> stones <0> killed <0> some <0> of <0> them
output: Score:3, Inappropriate Breaks: [].
example3:
input: they <0> made <0> a <0> choice <1> and <0> went <1> out <1> in <0> search <0> of <0> food
output: Score:3, Inappropriate Breaks: [“and <0> went”].
Example4:
input: the <0> following <1> week <3> the <0> same <0> policeman <0> sees <0> the <1> same <1> man <3> with <0> the <0> tiger <0> again
output: Score:2, Inappropriate Breaks: [“week <3> the”, “man <3> with”, “the <1> same”].
```
### few-shot for testset_3
```
Here are more examples that you can refer to:
example1:
input: but <0> the <0> mouse <0> finally <0> got <0> out <0> of <0> her <0> mouth <1> and <0> ran <0> away <2> since <2> she <3> could <0> not <0> bite <0> it
output: Score: 3, Inappropriate Breaks: [“she <3> could”].
example2:
input: our <0> class <1> is <0> going <0> to <0> put <1> on <0> a <0> show <0> on <0> may <0> seventh
output: Score:3, Inappropriate Breaks: [].
example3:
input: cinderella <2> danced <2> so <0> joyfully <3> that <0> she <0> forgot <3> the <0> time
output: Score:2, Inappropriate Breaks: [“cinderella <2> danced”, “danced <2> so”, “forgot <3> the”].
Example4:
input: the <0> following <0> week <3> the <0> same <0> policeman <0> sees <0> the <0> same <0> man <0> with <0> the <0> tiger <1> again
output: Score:3, Inappropriate Breaks: [“week <3> the”].
```
### few-shot for testset_4
```
Here are more examples that you can refer to:
example1:
input: our <0> class <1> is <0> going <0> to <0> put <1> on <0> a <0> show <1> on <0> may <0> seventh
output: Score: 3, Inappropriate Breaks:[].
example2:
input: our <0> class <1> is <0> going <0> to <0> put <0> on <1> a <0> show <1> on <0> may <0> seventh
output: Score:3, Inappropriate Breaks: [].
example3:
input: one <0> day <1> a <0> crow <0> stood <0> on <0> a <0> branch <2> near <0> his <0> nest <2> and <0> felt <0> very <0> happy <0> with <0> the <0> meat <0> in <0> his <0> mouth
output: Score:3, Inappropriate Breaks: [“branch <2> near].
Example4:
input: their <0> poor <0> parents <0> were <0> just <3> starving <2> as <0> usual
output: Score:3, Inappropriate Breaks: [“just <3> starving”].
```