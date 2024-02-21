from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  
import time
import socket
import threading
import os

# rea file from /content/Asia Voices Perspectives on Tax Policy Seminar 2022 - Panel Discussion.txt
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def split_text_into_chunks(text):
    """Split text into chunks below a maximum number of tokens.
    Actual GPT-3.5-turbo and PaLM2 models can take slightly more than this limit.
    Though we will stick to this for consistency.

    Args:
        text: The text to split.
        max_tokens: The maximum number of tokens per chunk.

    Returns:
        A list of text chunks, each with less than or equal to max_tokens.
    """

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=max_tokens,
        chunk_overlap=20, # overlap is usually done by the MapReduce method to improve performance
    )
    chunks = text_splitter.split_text(text)
    return chunks

class ChatGPTAutomation:

    """
    Adapted from https://github.com/Michelangelo27/chatgpt_selenium_automation/blob/master/handler/chatgpt_selenium_automation.py:
    1. Updated for support of latest selenium version
    2. Removed chrome_driver_path parameter
    3. Added getNextPage method
    """

    def __init__(self, chrome_path):
        """
        This constructor automates the following steps:
        1. Open a Chrome browser with remote debugging enabled at a specified URL.
        2. Prompt the user to complete the log-in/registration/human verification, if required.
        3. Connect a Selenium WebDriver to the browser instance after human verification is completed.

        :param chrome_path: file path to chrome.exe (ex. C:\\Users\\User\\...\\chromedriver.exe)
        :param chrome_driver_path: file path to chrome.exe (ex. C:\\Users\\User\\...\\chromedriver.exe)
        """

        self.chrome_path = chrome_path

        url = r"https://chat.openai.com"
        free_port = self.find_available_port()
        self.launch_chrome_with_remote_debugging(free_port, url)
        self.wait_for_human_verification()
        self.driver = self.setup_webdriver(free_port)



    def find_available_port(self):
        """ This function finds and returns an available port number on the local machine by creating a temporary
            socket, binding it to an ephemeral port, and then closing the socket. """

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]



    def launch_chrome_with_remote_debugging(self, port, url):
        """ Launches a new Chrome instance with remote debugging enabled on the specified port and navigates to the
            provided url """

        def open_chrome():
            chrome_cmd = f"{self.chrome_path} --remote-debugging-port={port} --user-data-dir=remote-profile {url}"
            os.system(chrome_cmd)

        chrome_thread = threading.Thread(target=open_chrome)
        chrome_thread.start()

    def setup_webdriver(self, port):
        """  Initializes a Selenium WebDriver instance, connected to an existing Chrome browser
             with remote debugging enabled on the specified port"""

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")

        # driver = webdriver.Chrome(executable_path=self.chrome_driver_path, options=chrome_options)
        # updated for support of latest selenium version
        driver = webdriver.Chrome(service=ChromeService(), options=chrome_options)
        return driver


    def send_prompt_to_chatgpt(self, prompt, add_prompt="TLDR in 200 words, without bullet points:"):
        """ Sends a message to ChatGPT and waits for 20 seconds for the response """

        input_box = self.driver.find_element(by=By.XPATH, value='//textarea[contains(@placeholder, "Message ChatGPT…")]')
        escaped_prompt = prompt.replace("\"", "")
        escaped_prompt = escaped_prompt.replace("\'", "\\'")

        # add "TLDR in 200 words, without bullet points" to front of prompt
        escaped_prompt = add_prompt + escaped_prompt

        # remove all next lines
        escaped_prompt = escaped_prompt.replace("\n", " ")

        print(escaped_prompt)

        self.driver.execute_script(f'arguments[0].value = "{escaped_prompt}";', input_box)
        input_box.send_keys(Keys.RETURN)
        input_box.submit()
        time.sleep(20)



    def return_chatgpt_conversation(self):
        """
        :return: returns a list of items, even items are the submitted questions (prompts) and odd items are chatgpt response
        """

        return self.driver.find_elements(by=By.CSS_SELECTOR, value='div.text-base')



    def save_conversation(self, file_name):
        """
        It saves the full chatgpt conversation of the tab open in chrome into a text file, with the following format:
            prompt: ...
            response: ...
            delimiter
            prompt: ...
            response: ...

        :param file_name: name of the file where you want to save
        """

        directory_name = "conversations"
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        delimiter = "|^_^|"
        chatgpt_conversation = self.return_chatgpt_conversation()
        with open(os.path.join(directory_name, file_name), "a") as file:
            for i in range(0, len(chatgpt_conversation), 2):
                file.write(
                    f"prompt: {chatgpt_conversation[i].text}\nresponse: {chatgpt_conversation[i + 1].text}\n\n{delimiter}\n\n")



    def return_last_response(self):
        """ :return: the text of the last chatgpt response """

        response_elements = self.driver.find_elements(by=By.CSS_SELECTOR, value='div.text-base')

        return response_elements[-1].text[7:].strip()



    def wait_for_human_verification(self):
        print("You need to manually complete the log-in or the human verification if required.")

        while True:
            user_input = input(
                "Enter 'y' if you have completed the log-in or the human verification, or 'n' to check again: ").lower()

            if user_input == 'y':
                print("Continuing with the automation process...")
                break
            elif user_input == 'n':
                print("Waiting for you to complete the human verification...")
                time.sleep(5)  # You can adjust the waiting time as needed
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    def getNextPage(self):
        # click //nav[@aria-label="Chat history"]//div[1]//a[1]
        # newChatBtn = self.driver.find_element(By.XPATH, "//nav[@aria-label=\"Chat history\"]//div[1]//a[1]")
        newChatBtn = self.driver.find_element(By.XPATH, "//div[@class='grow overflow-hidden text-ellipsis whitespace-nowrap text-sm text-token-text-primary' and text()='New chat']")
        newChatBtn.click()
        time.sleep(5)

    def quit(self):
        """ Closes the browser and terminates the WebDriver session."""
        print("Closing the browser...")
        self.driver.close()
        self.driver.quit()