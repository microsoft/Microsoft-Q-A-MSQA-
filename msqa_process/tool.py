import re
from langdetect import detect


HTTP_PAT = re.compile(r'''(?xi)
\b
(							# Capture 1: entire matched URL
  (?:
    https?:				# URL protocol and colon
    (?:
      /{1,3}						# 1-3 slashes
      |								#   or
      [a-z0-9%]						# Single letter or digit or '%'
      								# (Trying not to match e.g. "URI::Escape")
    )
    |							#   or
    							# looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)
    /
  )
  (?:							# One or more:
    [^\s()<>{}\[\]]+						# Run of non-space, non-()<>{}[]
    |								#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\)  # balanced parens, one level deep: (…(…)…)
    |
    \([^\s]+?\)							# balanced parens, non-recursive: (…)
  )+
  (?:							# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\)  # balanced parens, one level deep: (…(…)…)
    |
    \([^\s]+?\)							# balanced parens, non-recursive: (…)
    |									#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]		# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			# not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)
    \b
    /?
    (?!@)			# not succeeded by a @, avoid matching "foo.na" in "foo.na@example.com"
  )
)''')
USER_MENTION_PAT = re.compile(r'\[@.*?\]\(.*?\)') # 匹配开始，
PIC_PAT = re.compile(r"\!\[.*?]\[\d+\]")
REF_PAT = re.compile(r'\[\d+\]: /api/attachments/.*')
AZURE_LINK_PAT = re.compile(r"\[[^\[\]]*?\]\[\d+\]")
# AZURE_REF_PAT = re.compile(r'\[\d+\]:.*')
AZURE_REF_PAT = re.compile(r'\[\d+\]: \S+')
# SLASH_AND_DASH_PAT = re.compile(r"(\\-+|---+)")
SLASH_AND_DASH_PAT = re.compile(r"(\\--+|---+)")
SYMBOLS = " ,.·>\n!"
STAR_SYMBOL_PAT = re.compile(r"\*\*\*\*+")
EQUAL_PAT = re.compile(r"====+")
AZURE_ATTACHMENT_PAT= re.compile(r'/api/attachments/')

def pipeline(text, functions):
    for func in functions:
        text = func(text)
    return text

# def remove_user_mentions(text):
#     # pattern = re.compile(' \[@[^\]]+\]\(/users/na/\?userid=[^\)]+\)')
#     for x in USER_MENTION_PAT.findall(text):
#         text = text.replace(x, "")
#     return text


def remove_names_leq2words(text):
    # the first one or two words before first comma is name
    pattern = r'^([\w\s]+),'
    match = re.search(pattern, text)
    if match and len(match.group(1).strip().split()) <= 2:
        result = text[len(match.group(0)):]
    else:
        result = text
    # text = re.sub(pattern, '', text)
    last_line=result.split("\n")[-1]
    
    if len(last_line.split())<=3:
        result=result.replace(last_line,"")

    return result

def load_name_lists(p):
    with open(p, 'r') as f:
        name_list = [x.strip() for x in f.readlines()]
    return name_list

NAME_LIST = load_name_lists('username/name_all.txt')
def detect_name_and_remove(text):
    '''Suppose the text is clear at the begining and end.'''
    for name in NAME_LIST:
        pattern = r'(^|\b){}(\b|$)'.format(re.escape(name))
        if text.startswith(name) and re.match(pattern, text):
            text = text[len(name):]
        if text.endswith(name) and re.search(pattern, text):
            text = text[:-len(name)]
    return text

def detect_name_and_remove_improve(text):
    '''Suppose the text is clear at the begining and end.'''
    pattern = r'(^|\s)(?:{})($|\s)'.format('|'.join(map(re.escape, NAME_LIST)))
    text=re.sub(pattern, '', text)
    return text

def detect_and_remove_user_mentions(text):
    pat_list = [
        re.compile(r'\[@.*?\]\(.*?\)'), 
        re.compile(r'@\[.*?\]\(.*?\)'), 
        re.compile(r'\[.*?\]\(.*?userid=.*?\)'),
        re.compile(r'@\[.*?\]')
    ]

    results = []
    for pat in pat_list:
        res = pat.findall(text)
        results.extend(res)
    for x in results:
        text = text.replace(x, "")
    return text

def detect_and_remove_user_mentions_2(text):
    pat_list = [
        re.compile(r'\[@.*?\]\(.*?\)'), 
        re.compile(r'@\[.*?\]\(.*?\)'), 
        re.compile(r'\[.*?\]\(.*?userid=.*?\)'),
        re.compile(r'@\[.*?\]'),
        re.compile(r'@.*? ')
    ]
    results = []
    if text.startswith("@"):
        target = text.split('\n')[0]
        for pat in pat_list:
            res = pat.findall(target)
            results.extend(res)
        for x in results:
            text = text.replace(x, "")
    return text

def detect_welcome(text):
    for paragraph in text.split("\n"):
        if "welcome" in paragraph.lower():
            pat_list = ['Q&A', 'QnA', 'Q & A', 'Q&amp;A','Q@A']
            lower_pat_list = ['microsoft','welcome to azure','welcome back',
                                'welcome to our forum','you are welcome','you\'re welcome',
                                'welcome and thank you for the question',
                                'good day and welcome','your welcome']
            flag = False
            for pat in pat_list:
                flag |= pat in paragraph
            for pat in lower_pat_list:
                flag |= pat in paragraph
            return flag
        else:
            return False
        
def detect_and_remove_welcome(text):
    target_paragraphs = []
    for paragraph in text.split("\n"):
        if "welcome" in paragraph.lower():
            pat_list = ['Q&A', 'QnA', 'Q & A', 'Q&amp;A','Q@A']
            lower_pat_list = ['microsoft','welcome to azure','welcome back',
                                'welcome to our forum','you are welcome','you\'re welcome',
                                'welcome and thank you for the question',
                                'good day and welcome','your welcome']
            flag = False
            for pat in pat_list:
                flag |= pat in paragraph
            for pat in lower_pat_list:
                flag |= pat in paragraph.lower()
            if flag:
                target_paragraphs.append(paragraph)
    for paragraph in target_paragraphs:
            text = text.replace(paragraph, "")
    return text

    # split_pat = r'[.\n]'
    # if detect_welcome(text):
    #     target_paragraphs = []
    #     for paragraph in re.split(split_pat, text):
    #         if "welcome" in paragraph.lower():
    #             target_paragraphs.append(paragraph)
    #     for paragraph in target_paragraphs:
    #         text = text.replace(paragraph, "")
    # return text

def detect_and_remove_hello(text):
    split_pat_1 = r'[.\n\r]'
    split_pat_2 = r'([,!]|\s{2})'
    hi_pat = re.compile('^[h|H]i[ .*?,|,]')
    hello_pat = re.compile('^[h|H]ello[ .*?,|,]')
    target_paragraph = []
    for paragraph in re.split(split_pat_1, text):
        if len(hi_pat.findall(paragraph)) > 0:
            target_paragraph.append(re.split(split_pat_2, paragraph)[0])

        if len(hello_pat.findall(paragraph)) > 0:
            target_paragraph.append(re.split(split_pat_2, paragraph)[0])
    if len(target_paragraph) > 0:
        for paragraph in target_paragraph:  
            text = text.replace(paragraph, "")
    return text

def detect_and_remove_thank(text):
    pat_1 = re.compile("[t|T]hank.*?[,|.|\n|!|\r|-]")
    res_1 = pat_1.findall(text)
    if len(res_1) > 0:
        for target_paragragh in res_1:
            text = text.replace(target_paragragh, "")
    return text

def detect_and_remove_hope(text):
    hope_pat = re.compile("I?\s*[h|H]ope this helps?.*?[\n|\r|.|!|)]")
    res = hope_pat.findall(text)
    if len(res) > 0:
        for x in res:
            if "http" in x:continue
            text = text.replace(x, "")
    return text

def detect_and_remove_know(text):
    know_pat = [
        re.compile(r'Please let me know.*?[\n|\r|.|!|)]'),
        re.compile(r'Let me know.*?[\n|\r|.|!|)]'),
        re.compile(r'further.*?let me know.*?[\n|\r|.|!|)]'),
        re.compile(r'further.*?please let me know.*?[\n|\r|.|!|)]'),
        re.compile(r'Please let us know.*?[\n|\r|.|!|)]'),
        re.compile(r'Let us know.*?[\n|\r|.|!|)]'),
        re.compile(r'further.*?let us know.*?[\n|\r|.|!|)]'),
        re.compile(r'further.*?please let us know.*?[\n|\r|.|!|)]'),
    ]
    for pat in know_pat:
        results = pat.findall(text)
        results = [x for x in results if 'http' not in x]
        if len(results) > 0:
            for x in results:
                text = text.replace(x, "")
            break
    return text

def detect_and_remove_regards(text):
    split_pat = r"[.!\n\r]"
    results = []
    for sentence in re.split(split_pat, text):
        if ('regards' in sentence or 'Regards' in sentence) and len(sentence) <= 30:
            results.append(sentence)
    if len(results) > 0:
        for res in results:
            text = text.replace(res, "")
    return text

def replace_reference_with_link(text):
    az_link = [x.strip() for x in AZURE_LINK_PAT.findall(text)]
    az_ref = [x.strip() for x in AZURE_REF_PAT.findall(text)]
    num2link = {}
    for ref in az_ref:
        ref_num, link = ref.split(": ")
        num2link[ref_num] = link

    for link in az_link:
        description, ref_num = link.split("][")
        description = description.strip()+"]"
        ref_num = "[" + ref_num.strip()
        
        if ref_num in num2link:
            
            new_link = link.replace(ref_num, f"({num2link[ref_num]})")
            text = text.replace(link, new_link)
        else:continue
    return text

def plain_text_len(text):
    http_links = HTTP_PAT.findall(text)
    for x in http_links:
        text = text.replace(x, "<link>")
    
    pic_links = PIC_PAT.findall(text)
    for x in pic_links:
        text = text.replace(x, "<link>")
    
    ref_links = REF_PAT.findall(text)
    for x in ref_links:
        text = text.replace(x, "<link>")
    return len(text)

def is_too_short(text, min_len=100):
    l = plain_text_len(text)
    return (detect_link(text) and  l < min_len), l

def is_too_long(text, max_len=30000):
    l = plain_text_len(text)
    return (detect_link(text) and  l > max_len), l

# 1.

def detect_and_remove_all_below(text, detect_func):
    '''
    detect_func: {detect_user_mentions, detect_accept_answer}
    '''
    res = ""
    for text_piece in text.split("\n"):
        if not detect_func(text_piece):
            res += text_piece + "\n"
        else:
            break
    return res.strip("\n")

def remove_accept_answer_all_below(text):
    return detect_and_remove_all_below(text, detect_accept_answer)

# 1. 

def detect_and_remove_line(text, detect_func):
    '''
    detect_func: {detect_user_mentions, detect_accept_answer}
    '''
    res = ""
    # split_pat = r'[\n\r]'
    # for text_piece in re.split(split_pat,text):
    for text_piece in text.split("\n"):
        if not detect_func(text_piece):
            res += text_piece + "\n"
    return res.strip("\n")

def remove_user_mentions_line(text):
    return detect_and_remove_line(text, detect_user_mentions)

def remove_accept_answer_line(text):
    return detect_and_remove_line(text, detect_accept_answer)

def remove_email_notification_line(text):
    return detect_and_remove_line(text, detect_email_notification)

def remove_slash_and_dash_line(text):
    return detect_and_remove_line(text, detect_slash_and_dash)

def remove_equal_line(text):
    return detect_and_remove_line(text, detect_equal)

def remove_symbols_only_line(text):
    '''Dangerous: code block'''
    return detect_and_remove_line(text, detect_is_symbols_only)

def remove_ref_line(text):
    return detect_and_remove_line(text, detect_ref_link)

def remove_star_symbol_line(text):
    '''Dangerous: bold in markdown'''
    return detect_and_remove_line(text, detect_star_symbol)

def detect_user_mentions(text):
    if len(USER_MENTION_PAT.findall(text)) > 0: # detect user mention
        return True
    else:
        return False

def _detect_accept_answer_(text, keywords1=None, keywords2=None):
    text = text.lower()
    if keywords1 is None:
        keywords1 = ['answers','answer']
    if keywords2 is None:
        keywords2 = ['accept','accepts', 'replies as answers']
    if any(keyword in text for keyword in keywords1) and any(keyword in text for keyword in keywords2):
        res = True
    else:
        res = False
    return res

def detect_accept_answer(text):
    # old
    text = text.lower()
    if "accept as answer" in text \
        or "accept answer" in text \
        or "accept response" in text \
        or "accept the response" in text \
        or "accept the answer" in text \
        or "marked as answer" in text \
        or "up-vote" in text \
        or "upvote" in text \
        or "vote as helpful" in text \
        or "accept helpful response" in text \
        or "if you feel helpful" in text \
        or "replies as answer" in text \
        or "mark the answer" in text \
        or "mark the response" in text:
        return True
    else:
        return False

def detect_email_notification(text):
    text = text.lower()
    if 'email notification for this thread' in text:
        return True
    else:
        return False
    
def detect_is_symbols_only(text):
    text = text.strip(SYMBOLS)
    return len(text) == 0

    
def detect_slash_and_dash(text):
    '''
    1. \\----
    2. ---
    '''
    if len(SLASH_AND_DASH_PAT.findall(text)) > 0:
        return True
    else:
        return False

def detect_equal(text):
    '''
    1. ===
    '''
    if len(EQUAL_PAT.findall(text)) > 0:
        return True
    else:
        return False
    
def remove_multiple_n(text):
    text = text.replace("\r", "\n")
    return re.sub('\n+', '\n', text)

def remove_space(text):
    text = text.strip()
    text = text.lstrip(SYMBOLS)
    return text

def detect_star_symbol(text):
    if len(STAR_SYMBOL_PAT.findall(text)) > 0: # detect user mention
        return True
    else:
        return False


# 2.
def detect_http_link(text):
    urls = HTTP_PAT.findall(text)
    return len(urls) > 0

def detect_pic_link(text):
    res = PIC_PAT.findall(text)
    return len(res) > 0

def detect_ref_link(text):
    res = AZURE_REF_PAT.findall(text)
    return len(res) > 0

def detect_link(text):
    return detect_http_link(text) or detect_pic_link(text) or detect_ref_link(text)


# 3. remove case
def detect_and_remove_case(text, detect_func):
    if detect_func(text):
        return '1155121439'
    else:
        return text
    
def detect_attachment(text):
    return len(AZURE_ATTACHMENT_PAT.findall(text)) > 0
    
def detect_and_remove_pic_case(text):
    return detect_and_remove_case(text, detect_attachment)

def detect_and_remove_symbols_only_question(text):
    return detect_and_remove_case(text, detect_is_symbols_only)

def detect_is_en(text):
    '''
    True: Not English.
    False: English.
    '''
    try:
        lang = detect(text)
    except:
        print('Error in language detection.')
        return True
    else:
        return lang != 'en'

def detect_and_remove_not_en_question(text):
    return detect_and_remove_case(text, detect_is_en)