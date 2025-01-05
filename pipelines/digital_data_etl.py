from steps.etl import crawl_links, get_or_create_user

# first pipeline step that gets and creates the user and crawls the designated links in the user configuration file
@pipeline
def digital_data_etl(user_full_name: str, links:list[str]) -> str:
    user = get_or_create_user(user_full_name)
    last_step = crawl_links(user=user, links = links)

    return last_step.invocation_id