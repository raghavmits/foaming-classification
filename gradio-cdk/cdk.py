import os
from pathlib import Path
from constructs import Construct
from aws_cdk import App, Stack, Environment, Duration, CfnOutput
from aws_cdk import aws_iam as iam
from aws_cdk.aws_lambda import DockerImageFunction, DockerImageCode, Architecture, FunctionUrlAuthType, HttpMethod
from aws_cdk.aws_logs import RetentionDays

my_environment = Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION'))


class GradioLambda(Stack):
    def __init__(self, scope: Construct, construct_id: str, target_architecture="arm", **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        architecture = Architecture.ARM_64 if target_architecture == "arm" else Architecture.X86_64

        # create function
        lambda_fn = DockerImageFunction(
            self,
            "GradioFunction",
            code=DockerImageCode.from_image_asset(str(Path.cwd()), file="Dockerfile"),
            architecture=Architecture.X86_64,
            memory_size=4096,
            timeout=Duration.minutes(2),
        )
        # add HTTPS url
        fn_url = lambda_fn.add_function_url(
                    auth_type=FunctionUrlAuthType.NONE,
                    cors={
                        "allowed_methods": [HttpMethod.ALL],
                        "allowed_headers": ["*"],
                        "allowed_origins": ["*"]
                    }
                )
        CfnOutput(self, "functionUrl", value=fn_url.url, description="URL for the Gradio interface")



app = App()
gradio_lambda = GradioLambda(app, "GradioLambda", env=my_environment)

app.synth()
