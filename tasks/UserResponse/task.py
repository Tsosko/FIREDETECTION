[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph

accept_string    = variables.get("accept_string")
reject_string    = variables.get("reject_string")
case_sensitive   = variables.get("case_sensitive") == "true"
default_response = variables.get("default_response")

class UserResponse:
    def __init__(self):
        pass

    def check_user_response(self, user_response: str) -> bool:
        resp = user_response if case_sensitive else user_response.lower()
        accept = accept_string if case_sensitive else accept_string.lower()
        reject = reject_string if case_sensitive else reject_string.lower()

        if resp == accept:
            return True
        if resp == reject:
            return False
        if default_response == "accept":
            return True
        if default_response == "reject":
            return False
        return False

if __name__ == "__main__":
    transmittedmsg = "yes"
    if isinstance(transmittedmsg, dict) and 'user_response' in transmittedmsg:
        user_resp = transmittedmsg['user_response']
    elif isinstance(transmittedmsg, str):
        user_resp = transmittedmsg
    else:
        user_resp = default_response

    manager = UserResponse()
    accepted = manager.check_user_response(user_resp)

    resultMap.put("USER_RESPONSE_ACCEPTED", str(accepted))