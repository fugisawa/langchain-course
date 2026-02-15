import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Linus Benedict Torvalds[a] (born 28 December 1969) is a Finnish and American software
    engineer who is the creator and lead developer of the Linux kernel since 1991.
    He also created the distributed version control system Git.

    Torvalds was one of the recipients of the 2012 Millennium Technology Prize
    "in recognition of his creation of a new open source operating system for
    computers leading to the widely used Linux kernel".[4] He is also the recipient
    of the 2014 IEEE Computer Society Computer Pioneer Award[5] and the 2018 IEEE
    Masaru Ibuka Consumer Electronics Award.
    """

    summary_template = """
    Given the information {information} about a person, provide a summary of the text, containing:
    1. A short summary of the text.
    2. Two interesting facts about the this person.
    """

    summary_prompt = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatGoogleGenerativeAI(google_api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3-pro-preview")
    
    chain = summary_prompt | llm
    
    res = chain.invoke(input={"information": information})
    print(res.content)
    
if __name__ == "__main__":
    main()
