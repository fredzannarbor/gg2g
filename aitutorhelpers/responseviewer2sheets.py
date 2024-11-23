import re

text = """
All Responses
Response 1
Borg's scale used the premise that individuals could introspectively evaluate their physical state during exercise, and these evaluations were mapped onto a numerical scale. Specifically, the original Borg scale ranged from 6 to 20, with the numbers chosen to roughly correspond to the heart rate of a healthy adult. For example, a rating of 6 was meant to align with a heart rate of about 60 beats per minute (typical resting heart rate), while a rating of 20 aligned with a heart rate of about 200 beats per minute (approximate maximal heart rate for a young adult).

This numerical mapping allowed individuals to self-regulate exercise intensity and communicate that intensity to others. The scale was meant to give a subjective measure of exertion based on how an individual feels in terms of heart rate and physical effort.

Words: 137 · Characters: 828 · Paragraphs: 2
Response 2
Borg's Scale used the concept of perceived exertion to allow individuals to introspectively evaluate their physical state during exercise and consistently map these evaluations onto a numerical scale. The numbers for the original Borg scale (6 to 20) were chosen to roughly correspond with a healthy adult's heart rate:

A rating of 6 corresponds to a heart rate of about 60 beats per minute, which is the resting heart rate for many people.
A rating of 20 corresponds to a heart rate of about 200 beats per minute, which is close to the maximal heart rate of a young adult.
This precise calibration allowed for self-regulation of exercise intensity and facilitated communication of that intensity to others, such as coaches or healthcare providers.

Additionally, the revised Borg CR10 Scale ranges from 0 to 10, with verbal anchors at each level (such as "moderate," "strong," "very strong") to help users better gauge their level of exertion.

In essence, individuals select the number on Borg’s scale that best matches their perceived effort or exertion during the exercise based on these numerical and perceptual cues.

Words: 184 · Characters: 1130 · Paragraphs: 5
Response 3
Borg's scale used individuals' introspective evaluations of their physical state during exercise to determine the appropriate number for their Rating of Perceived Exertion (RPE). The scale was designed to allow people to consistently map their self-evaluated exertion levels onto a numerical scale.

The original Borg scale ranged from 6 to 20, with specific numbers chosen to roughly correspond to the heart rate values of a healthy adult (e.g., a rating of 6 correlating with about 60 beats per minute, and 20 with 200 beats per minute). This meant that individuals could gauge their level of exertion based on their physical sensations and perceive how hard they were working in terms of an easily understandable numerical representation.

The later Borg CR10 Scale, which ranged from 0 to 10, included verbal anchors to further assist users in gauging their exertion levels more accurately. However, while the scale can assess perceived effort during resistance training, its original focus on endurance exercise means it may not always be the best tool for determining proximity to muscle failure, especially in high-repetition scenarios.

Words: 177 · Characters: 1144 · Paragraphs: 3
Response 4
Borg's Scale uses individuals' introspective evaluations of their physical state during exercise to gauge the RPE (Rate of Perceived Exertion) of an exercise. The numbers chosen for the original Borg Scale, which ranges from 6 to 20, are meant to correspond roughly to the heart rate of a healthy adult, with 6 indicating "no exertion" and corresponding to a heart rate of about 60 beats per minute (typically a resting heart rate), and 20 indicating "maximal exertion" and corresponding to a heart rate of about 200 beats per minute (a maximal heart rate for a young adult). This numerical system provided a way for individuals to self-regulate exercise intensity based on their perceived effort and communicate it effectively to others, such as coaches or healthcare providers. The scale's design facilitates a bridge between subjective feeling and a quantifiable number, linking perceived exertion with physiological markers like heart rate.

Words: 148 · Characters: 944 · Paragraphs: 1
Response 5
According to the text, Borg's original scale (introduced in 1962) used heart rate as a basis for gauging which number to select for the Rating of Perceived Exertion (RPE) of an exercise. The scale ranged from 6 to 20, with these numbers chosen to roughly correspond with the heart rate of a healthy adult:

A rating of 6 corresponded to a heart rate of about 60 beats per minute (typical resting heart rate for many people).

A rating of 20 corresponded to a heart rate of 200 beats per minute (the maximal heart rate of a young adult).

This correlation between the scale numbers and heart rate was intended to provide a simple way for individuals to estimate their level of exertion during exercise, particularly in endurance-based activities. However, it's important to note that this scale was later revised, and its application to resistance training may have limitations, as mentioned in the last paragraph of the text.

Words: 159 · Characters: 931 · Paragraphs: 4
Response 6
Borg's scale, which comes in two versions, relies on an individual's introspective evaluation of their physical state during exercise to gauge the level of exertion. For the original Borg scale (1962) that ranges from 6 to 20, the numbers were chosen to roughly correspond with the heart rate of a healthy adult: a rating of 6 corresponds to a heart rate of about 60 beats per minute, and a rating of 20 corresponds to 200 beats per minute. The Borg CR10 Scale (1982), which ranges from 0 to 10, includes verbal anchors at each level (e.g., moderate, strong, very strong) to help users better gauge their level of exertion. These verbal anchors allow for more consistent mapping of the introspective evaluation onto the numerical scale, enabling users to communicate their intensity more effectively to coaches or healthcare providers.

Words: 138 · Characters: 835 · Paragraphs: 1
Response 7
Borg's Scale was designed for individuals to introspectively evaluate their physical state during exercise and consistently map these evaluations onto a numerical scale. The original Borg Scale, introduced in 1962, used a 6 to 20 range, with each number roughly corresponding to the expected heart rate of a healthy adult (a rating of 6 corresponding to a heart rate of about 60 beats per minute and a rating of 20 corresponding to 200 beats per minute). This allowed for self-regulation of exercise intensity and communication of that intensity to others.

In 1982, Borg introduced the Borg CR10 Scale, a revised version with a scale from 0 ("nothing at all") to 10 ("extremely strong"). This scale included verbal anchors at each level (e.g., moderate, strong, very strong) to help users better gauge their level of exertion. These verbal anchors facilitated more accurate and consistent self-assessment of perceived exertion during exercise.

Words: 149 · Characters: 944 · Paragraphs: 2
Response 8
Borg's Scale used subjective feelings of exertion to gauge which number to select for the Rating of Perceived Exertion (RPE) of an exercise. This includes:

Introspective evaluation of one's physical state during exercise.
Physical sensations such as breathlessness, discomfort, strain, fatigue, and heart rate perception.
The original scale from 6 to 20 was correlated with the heart rate of a healthy adult, assuming that individuals could align their perception of effort with a range where the physical exertion would roughly match the heart rate in beats per minute. The revised Borg CR10 Scale uses verbal anchors along with numbers to describe these sensations, ranging from "nothing at all" to "extremely strong," making it easier for individuals to select a number that best represents their level of exertion based on how they feel during the activity.

Words: 137 · Characters: 879 · Paragraphs: 3
Response 9
Borg's scale, specifically in its original form from 6 to 20, used heart rate as a proxy to help individuals determine which number to select to represent their Rating of Perceived Exertion (RPE). The idea was that the exertion level could be inferred from the heart rate, where a score of 6 was intended to correspond to a heart rate of 60 beats per minute, typical of rest, and a score of 20 to a heart rate of 200 beats per minute, which would be near maximal effort for many young adults. Additionally, with the Borg CR10 Scale, there are verbal anchors at each level to help users better gauge their level of exertion by relating sensations to descriptive words like "moderate," "strong," or "very strong."

Words: 126 · Characters: 719 · Paragraphs: 1
Response 10
Borg's scale used perceived exertion to gauge which number to select for the RPE (Rate of Perceived Exertion) of an exercise. This means individuals would introspectively assess how hard they feel they are working during exercise, taking into account factors like physical sensations, fatigue, and breathlessness. These subjective feelings of exertion are then matched to a numerical value on the scale, which in the original 6-20 scale, was designed to approximately align with the heart rate of the exercising individual.

Words: 80 · Characters: 527 · Paragraphs: 1
Response 11
Borg's original scale used heart rate as a gauge for selecting the RPE (Rate of Perceived Exertion) number during exercise. The scale ranged from 6 to 20, where a rating of 6 corresponded to a heart rate of about 60 beats per minute (resting heart rate for many people), and a rating of 20 corresponded to 200 beats per minute, which is close to the maximal heart rate of a young adult. This allowed individuals to relate their subjective feelings of exertion to an objective measure of heart rate, making it easier to consistently gauge and communicate exercise intensity.

Words: 99 · Characters: 573 · Paragraphs: 1
Response 12
Borg's scale, originally introduced in 1962, used the concept of a numerical rating that corresponded to the heart rate of a healthy adult. The scale ranged from 6 ("no exertion") to 20 ("maximal exertion"), with the numbers chosen so that a rating of 6 roughly corresponded to a heart rate of around 60 beats per minute (similar to a resting heart rate for many individuals) and a rating of 20 corresponded to around 200 beats per minute (an estimate of the maximal heart rate for a young adult). This correlation with heart rate aimed to provide a consistent measure of perceived exertion during physical activity.

As for the Borg CR10 Scale introduced in 1982, it's a revision of the original scale and ranges from 0 ("nothing at all") to 10 ("extremely strong"). It includes verbal anchors at each level to help individuals gauge their level of exertion more effectively. This scale still relies on the general principle of self-assessed physical state, but it doesn't directly translate to heart rate measurements like the original scale did; rather, it relies on the user's subjective feelings of effort and fatigue.

Words: 187 · Characters: 1123 · Paragraphs: 2
Response 13
Borg's original scale used heart rate as a rough guide to select the RPE (Rate of Perceived Exertion) of an exercise, where the range from 6 to 20 on the scale corresponded approximately to heart rates from 60 to 200 beats per minute.

Words: 43 · Characters: 234 · Paragraphs: 1
Response 14
Borg's scale uses the individual's perception of their exertion level and physical state during exercise to gauge which number to select as the RPE (Rating of Perceived Exertion). For the original Borg scale (6 to 20), the numbers were chosen to roughly correspond with heart rate, thus aligning perceived effort with actual physiological output like heart beats per minute. In the revised Borg CR10 Scale, the scale ranged from 0 ("nothing at all") to 10 ("extremely strong"), with verbal anchors at each level to help users better quantify their level of exertion based on how they feel, rather than directly tying it to a physiological measure like heart rate. However, even if the scale was not specifically designed for resistance training or gauging proximity to failure, the idea is that individuals would use their subjective experience of effort to choose an appropriate number on the scale.

Words: 146 · Characters: 900 · Paragraphs: 1
Response 15
Borg's Scale used the perception of physical exertion and effort as felt by the individual during exercise to determine which number to select for the RPE (Rate of Perceived Exertion). In the original scale, introduced in 1962, the numbers were selected to correspond roughly with the heart rate of a healthy adult, making the assumption that perceived exertion would increase linearly with heart rate. Here's a brief summary:

1962 Borg Scale: Ranges from 6 ("no exertion") to 20 ("maximal exertion"). The scale was designed so that a rating of 6 might correspond to a heart rate of 60 beats per minute, and 20 to about 200 beats per minute, which are typical resting and maximal heart rates, respectively.

1982 Borg CR10 Scale: This revised version, the Category-Ratio (CR) Scale, went from 0 ("nothing at all") to 10 ("extremely strong"). This scale included verbal anchors to help users better quantify their level of exertion based on feeling rather than direct correlation with heart rate.

Words: 165 · Characters: 1008 · Paragraphs: 3
Response 16
Borg's original scale used heart rate to help gauge which number to select for the RPE (Rate of Perceived Exertion) of an exercise. Here's how it was structured:

A rating of 6 corresponded roughly to a resting heart rate of about 60 beats per minute.
A rating of 20 was meant to correlate with a maximal heart rate of about 200 beats per minute for a young adult.
This design allowed users to somewhat align their exertion level with a physiological response (heart rate), providing a more objective measure for what is generally a subjective feeling of effort. However, this correlation with heart rate has some limitations, particularly when applying the scale to resistance training or different conditions affecting heart rate or perceived exertion.

Words: 126 · Characters: 772 · Paragraphs: 3
"""

import re

def transform_to_google_sheets_format(text):
    lines = text.split('\n')
    responses = {}
    current_response = None

    for line in lines:
        print(line)
        if line.startswith("Words") or line.startswith("Response"):
            continue

        match = re.match(r'^Response (\d+)$', line)
        if match:
            current_response = int(match.group(1))
            responses[current_response] = ""  # Initialize with an empty string
        elif current_response is not None:
            # Always add a space before appending, handling empty lines correctly
            responses[current_response] += (" " if responses[current_response] else "") + line.strip()
            print(responses[current_response])
    google_sheet_format = "Response #\tText\n"
    for i in range(1, 17):
        google_sheet_format += f"{i}\t{responses.get(i, '')}\n"  # Use .get() to handle missing responses

    return google_sheet_format

formatted_output = transform_to_google_sheets_format(text)
print(formatted_output)

# If you want to save this to a file:
with open('output.tsv', 'w', encoding='utf-8') as file:
    file.write(formatted_output)